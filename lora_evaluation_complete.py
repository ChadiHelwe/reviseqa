#!/usr/bin/env python3
"""
Complete LoRA Model Evaluation Script
Matches the exact evaluation methodology from src/evaluation.py

Evaluates LoRA finetuned models on logical reasoning datasets with:
- Multiple tracks (implicit, explicit, with/without reasoning, with/without correction)
- Detailed per-task JSON output
- Token count tracking
- Performance metrics (accuracy, degradation, step-by-step, etc.)
- Compatible with original evaluation pipeline
"""

import csv
import json
import time
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from argparse import ArgumentParser
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import tiktoken


def extract_answer_from_prediction(prediction):
    """
    Extract the answer from a prediction string that may contain JSON with reasoning and answer.
    """
    if not isinstance(prediction, str):
        return str(prediction), ""

    # First try to parse as direct JSON
    try:
        parsed = json.loads(prediction)
        if isinstance(parsed, dict) and 'answer' in parsed:
            answer = parsed.get('answer', '')
            reasoning = parsed.get('reasoning', '')
            return str(answer), str(reasoning)
    except (json.JSONDecodeError, TypeError):
        pass

    # Try to find JSON block within the prediction
    json_patterns = [
        r'\{[^{}]*"reasoning"[^{}]*"answer"[^{}]*\}',
        r'\{.*?"reasoning".*?"answer".*?\}',
        r'\{[^{}]*"answer"[^{}]*\}',
    ]

    for pattern in json_patterns:
        json_match = re.search(pattern, prediction, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict) and 'answer' in parsed:
                    answer = parsed.get('answer', '')
                    reasoning = parsed.get('reasoning', '')
                    return str(answer), str(reasoning)
            except json.JSONDecodeError:
                continue

    # Plain text extraction
    extracted_answer = ""
    extracted_reasoning = ""

    # Extract reasoning
    reasoning_patterns = [
        r'Reasoning:\s*(.*?)(?=Answer:|answer:|$)',
        r'reasoning:\s*(.*?)(?=Answer:|answer:|$)',
        r'"reasoning":\s*"([^"]*)"',
    ]
    for pattern in reasoning_patterns:
        match = re.search(pattern, prediction, re.DOTALL | re.IGNORECASE)
        if match:
            extracted_reasoning = match.group(1).strip()
            break

    # Extract answer
    answer_patterns = [
        r'Answer:\s*(True|False|Uncertain|true|false|uncertain)',
        r'answer:\s*(True|False|Uncertain|true|false|uncertain)',
        r'"answer":\s*"([^"]*)"',
        r'\b(True|False|Uncertain)\b',
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, prediction, re.IGNORECASE)
        if match:
            extracted_answer = match.group(1).strip()
            break

    if extracted_answer.lower() in ['true', 'false', 'uncertain']:
        extracted_answer = extracted_answer.capitalize()

    if not extracted_answer:
        extracted_answer = prediction.strip()

    return extracted_answer, extracted_reasoning


PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Options:
A) True
B) False
C) Uncertain

"""

ANSWER_EXAMPLE = """The correct option is: {{
    "reasoning": {reasoning},
    "answer": {answer}
}}"""

CORRECTION = """You made a mistake, the correct answer was: {correct_answer}. Now answer the next problem.
{context}"""


@dataclass
class LogicData:
    context: str
    question: str
    answer: str
    reasoning: str = None
    tags: List[str] = field(default_factory=list)


@dataclass
class LogicDataset:
    explicit_data: List[List[LogicData]] = field(default_factory=list)
    implicit_data: List[List[LogicData]] = field(default_factory=list)
    implicit_shuffled_data: List[List[LogicData]] = field(default_factory=list)
    filenames: List[str] = field(default_factory=list)

    def read_dir(self, data_dir: str, enable_truncated: bool = False) -> None:
        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith('.json'):
                continue
            if not enable_truncated and fname.endswith("_truncated.json"):
                continue
            path = os.path.join(data_dir, fname)
            with open(path, 'r') as f:
                data = json.load(f)

            original_context = "\n".join(data.get("original_context", []))
            reasoning_steps = []
            for step in data.get("reasoning_chain", []):
                facts = " ".join(f["text"] for f in step.get("facts", []))
                rules_list = step.get("rules", [])
                rule = rules_list[0].get("text", "") if rules_list else ""
                concl = (step.get("conclusion") or {}).get("text", "")
                reasoning_steps.append(f"{facts}. {rule}. Therefore, {concl}.")
            reasoning_text = " ".join(reasoning_steps)

            base_concl = data.get("conclusion", "")
            question_demo = f"Does the context entail the conclusion '{base_concl}'?"
            edits = data.get("edits", [])
            demo_answer = edits[-1].get("answer") if edits else data.get("answer")

            # Original sample
            i0 = LogicData(
                context=original_context,
                question=question_demo,
                answer=demo_answer,
                reasoning=reasoning_text,
                tags=["original"],
            )

            implicit_chain = [i0]
            explicit_chain = [i0]

            for edit in edits:
                imp_ctx = "\n".join(edit.get("edited_natural_language_context", []))
                imp_concl = edit.get("conclusion", "")
                imp_q = f"Does the context entail the conclusion '{imp_concl}'?"
                imp_a = edit.get("answer", "")

                # Determine tags
                delta = edit.get("edits_made", {})
                tags = []
                if delta.get("removed_facts"): tags.append("removed_facts")
                if delta.get("removed_rules"): tags.append("removed_rules")
                if delta.get("added_rules"): tags.append("added_rules")
                if delta.get("added_facts"): tags.append("added_facts")
                if not tags: tags = ["no_change"]

                implicit_chain.append(
                    LogicData(context=imp_ctx, question=imp_q, answer=imp_a, tags=tags)
                )

                # Build explicit context edits breakdown
                parts = []
                if delta.get("removed_facts"):
                    parts.append(
                        "Removed facts:\n" + "\n".join(f"- {f['nl']}" for f in delta["removed_facts"])
                    )
                if delta.get("removed_rules"):
                    parts.append(
                        "Removed rules:\n" + "\n".join(f"- {r['nl']}" for r in delta["removed_rules"])
                    )
                if delta.get("added_rules"):
                    parts.append(
                        "Added rules:\n" + "\n".join(f"- {r['nl']}" for r in delta["added_rules"])
                    )
                if delta.get("added_facts"):
                    parts.append(
                        "Added facts:\n" + "\n".join(f"- {f['nl']}" for f in delta["added_facts"])
                    )
                exp_ctx = "\n\n".join(parts)

                explicit_chain.append(
                    LogicData(context=exp_ctx, question=imp_q, answer=imp_a, tags=tags)
                )

            # Shuffled implicit
            shuffled_chain = []
            for entry in implicit_chain:
                sentences = [s for s in entry.context.replace('\n', ' ').split('. ') if s]
                random.shuffle(sentences)
                shuffled_ctx = '. '.join(sentences)
                shuffled_chain.append(
                    LogicData(
                        context=shuffled_ctx,
                        question=entry.question,
                        answer=entry.answer,
                        reasoning=entry.reasoning,
                        tags=entry.tags,
                    )
                )

            self.implicit_data.append(implicit_chain)
            self.explicit_data.append(explicit_chain)
            self.implicit_shuffled_data.append(shuffled_chain)
            self.filenames.append(fname[:-5] if fname.endswith('.json') else fname)

    def __len__(self):
        return len(self.explicit_data)


class LoRAConversation:
    """Conversation handler for LoRA finetuned models"""

    def __init__(
        self,
        base_model_name: str,
        lora_model_path: str,
        use_4bit: bool = True,
        use_8bit: bool = False,
        merge_weights: bool = False,
        max_new_tokens: int = 1024,
    ) -> None:
        self.base_model_name = base_model_name
        self.lora_model_path = lora_model_path
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None
        self.messages = []

        self._load_model(use_4bit, use_8bit, merge_weights)

    def _load_model(self, use_4bit, use_8bit, merge_weights):
        """Load LoRA model"""
        print(f"Loading base model: {self.base_model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Setup quantization
        quantization_config = None
        if not merge_weights:
            if use_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            elif use_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load base model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            **model_kwargs
        )

        # Load LoRA weights
        print(f"Loading LoRA weights: {self.lora_model_path}")
        self.model = PeftModel.from_pretrained(base_model, self.lora_model_path)

        if merge_weights:
            print("Merging weights...")
            self.model = self.model.merge_and_unload()

        self.model.eval()
        print("Model loaded successfully")

    def init_conversation(
        self, context: str, question: str, reasoning: str, answer: str
    ) -> None:
        """Initialize conversation with first example"""
        self.messages = []
        self.messages += [
            {
                "role": "system",
                "content": (
                    "When you reply, output *only* a JSON object with exactly "
                    "three fields:\n"
                    "  - reasoning  (a string)\n"
                    "  - answer     (one of 'True','False','Uncertain')\n"
                    "Do not wrap it in markdown, do not say anything else."
                ),
            },
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(context=context, question=question),
            },
            {
                "role": "assistant",
                "content": ANSWER_EXAMPLE.format(reasoning=reasoning, answer=answer),
            },
        ]

    def _messages_to_prompt(self):
        """Convert messages to prompt string"""
        # Try to use chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
            try:
                prompt = self.tokenizer.apply_chat_template(
                    self.messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return prompt
            except:
                pass

        # Fallback manual formatting
        prompt_parts = []
        for msg in self.messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                prompt_parts.append(f"### System:\n{content}\n")
            elif role == 'user':
                prompt_parts.append(f"### User:\n{content}\n")
            elif role == 'assistant':
                prompt_parts.append(f"### Assistant:\n{content}\n")

        return "\n".join(prompt_parts) + "### Assistant:\n"

    def send_request(self, role: str, content: str):
        """Send request to model"""
        self.messages.append({"role": role, "content": content})

        try:
            prompt = self._messages_to_prompt()

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            self.messages.append({"role": "assistant", "content": response})
            return response.strip()

        except Exception as e:
            print(f"Error during model generation: {e}")
            error_response = json.dumps({"reasoning": "ERROR", "answer": "ERROR"})
            self.messages.append({"role": "assistant", "content": error_response})
            return error_response


def _evaluate_batch(args):
    """Evaluate a single batch (chain) of examples"""
    path, start_idx, batch_entries, base_model, lora_model, use_4bit = args
    batch_scores = []
    token_counts = []
    step_records: List[Dict[str, Any]] = []
    detailed_predictions: List[Dict[str, Any]] = []
    length = 1
    mistake = False
    prev_correct = True
    recovery_count = 0

    enc = tiktoken.get_encoding("cl100k_base")
    conv = LoRAConversation(
        base_model_name=base_model,
        lora_model_path=lora_model,
        use_4bit=use_4bit,
        max_new_tokens=1024
    )

    include_reasoning = not "_no_reasoning" in path
    include_correction = not "_no_correction" in path

    first = batch_entries[0]
    conv.init_conversation(
        context=first.context,
        question=first.question,
        reasoning=(first.reasoning if include_reasoning else ""),
        answer=first.answer,
    )

    # Record first step (demonstration)
    detailed_predictions.append({
        "step": 0,
        "context": first.context,
        "question": first.question,
        "prediction": first.answer,
        "correct_answer": first.answer,
        "reasoning": first.reasoning if include_reasoning else "",
        "correct": True,
        "tags": first.tags,
        "is_demonstration": True
    })

    for step_idx, entry in enumerate(batch_entries[1:], 1):
        if prev_correct or not include_correction:
            ctx = entry.context
        else:
            ctx = CORRECTION.format(
                correct_answer=batch_entries[step_idx - 1].answer,
                context=entry.context,
            )
        prompt = PROMPT_TEMPLATE.format(context=ctx, question=entry.question)

        # Token count
        token_count_history = sum(len(enc.encode(msg["content"])) for msg in conv.messages)
        token_counts.append(token_count_history)

        response = conv.send_request(role="user", content=prompt)

        # Extract prediction
        try:
            parsed = json.loads(response)
            predicted = parsed.get("answer", "")
            reasoning_output = parsed.get("reasoning", "")
            correct_flag = int(entry.answer == predicted)
        except (json.JSONDecodeError, TypeError):
            predicted, reasoning_output = extract_answer_from_prediction(response)
            correct_flag = int(entry.answer == predicted)

        batch_scores.append(correct_flag)

        # Record detailed prediction
        detailed_predictions.append({
            "step": step_idx,
            "context": entry.context,
            "question": entry.question,
            "prediction": predicted,
            "correct_answer": entry.answer,
            "reasoning": reasoning_output,
            "correct": bool(correct_flag),
            "tags": entry.tags,
            "is_demonstration": False,
            "token_count": token_count_history
        })

        # Record step
        step_records.append({
            "chain_idx": start_idx,
            "step": step_idx,
            "token_count": token_count_history,
            "correct": correct_flag,
            "tags": entry.tags,
            "prediction": predicted,
            "correct_answer": entry.answer,
            "reasoning": reasoning_output,
        })

        if correct_flag and not prev_correct:
            recovery_count += 1
        if correct_flag and not mistake:
            length += 1
            prev_correct = True
        elif not correct_flag:
            prev_correct = False
            mistake = True

    return path, start_idx, batch_scores, length, step_records, detailed_predictions


class LoRAEvaluator:
    """Evaluator for LoRA finetuned models"""

    def __init__(
        self,
        dataset: LogicDataset,
        base_model: str,
        lora_model: str,
        batch_size: int = 1,
        use_4bit: bool = True,
        shuffled: bool = False,
        detailed_output_dir: str = None
    ) -> None:
        self.dataset = dataset
        self.base_model = base_model
        self.lora_model = lora_model
        self.batch_size = batch_size
        self.use_4bit = use_4bit
        self.detailed_output_dir = detailed_output_dir

        self.tracks = [
            "implicit",
            "explicit",
            "implicit_no_reasoning",
            "explicit_no_reasoning",
            "implicit_no_correction",
            "explicit_no_correction",
            "implicit_no_reasoning_no_correction",
            "explicit_no_reasoning_no_correction",
        ]
        if shuffled:
            self.tracks.append("implicit_shuffled")
            self.tracks.append("implicit_shuffled_no_reasoning")

        self.tally_score_per_prompt = {t: [[] for _ in range(len(self.dataset))] for t in self.tracks}
        self.length_score_per_prompt = {t: [0]*len(self.dataset) for t in self.tracks}
        self.token_stats: Dict[str, List[Dict[str, Any]]] = {t: [] for t in self.tracks}

        # Create detailed output directories
        if self.detailed_output_dir:
            os.makedirs(self.detailed_output_dir, exist_ok=True)
            for track in self.tracks:
                os.makedirs(os.path.join(self.detailed_output_dir, track), exist_ok=True)

    def return_tally_sum(self) -> Dict[str, int]:
        return {path: sum(sum(chain) for chain in scores) for path, scores in self.tally_score_per_prompt.items()}

    def return_length_by_difficulty(self) -> Dict[str, Dict[str, int]]:
        by_diff: Dict[str, Dict[str, int]] = {}
        for path, lengths in self.length_score_per_prompt.items():
            if path.startswith("explicit"):
                chains = self.dataset.explicit_data
            elif path.startswith("implicit_shuffled"):
                chains = self.dataset.implicit_shuffled_data
            else:
                chains = self.dataset.implicit_data
            easy = medium = hard = 0
            for idx, L in enumerate(lengths):
                total_steps = len(chains[idx])
                ratio = L / total_steps if total_steps > 0 else 0
                if ratio >= 1.0: hard += 1
                if ratio >= 0.6: medium += 1
                if ratio >= 0.3: easy += 1
            by_diff[path] = {"easy": easy, "medium": medium, "hard": hard}
        return by_diff

    def save_metrics(self, model_name: str, base_dir: str = "results") -> tuple:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        tally = self.return_tally_sum()
        by_diff = self.return_length_by_difficulty()

        # Degradation buckets
        bucket_size = 512
        degradation_buckets: Dict[str, Dict[str, Dict[str, int]]] = {}
        for track in self.tracks:
            bucket_totals: Dict[int, int] = {}
            bucket_corrects: Dict[int, int] = {}
            for rec in self.token_stats[track]:
                bucket = rec["token_count"] // bucket_size
                bucket_totals[bucket] = bucket_totals.get(bucket, 0) + 1
                if rec["correct"]:
                    bucket_corrects[bucket] = bucket_corrects.get(bucket, 0) + 1
            degradation_buckets[track] = {
                str(b): {"total": bucket_totals[b], "correct": bucket_corrects.get(b, 0)}
                for b in sorted(bucket_totals)
            }

        # Permutation stats
        perm_stats: Dict[str, Dict[str, Dict[str, int]]] = {t: {} for t in self.tracks}
        for track in self.tracks:
            for rec in self.token_stats[track]:
                for tag in rec.get("tags", ["untagged"]):
                    ps = perm_stats[track].setdefault(tag, {"total":0, "correct":0})
                    ps["total"] += 1
                    ps["correct"] += rec["correct"]

        correctness = {
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "base_model": self.base_model,
                "lora_model": self.lora_model,
                "dataset_length": len(self.dataset),
                "batch_size": self.batch_size
            },
            "total_per_track": {t: len(self.token_stats[t]) for t in self.tracks},
            "tally_sum": tally,
            "length_by_difficulty": by_diff,
            "degradation_buckets": degradation_buckets,
            "permutation_stats": perm_stats,
        }

        json_path = os.path.join(base_dir, f"lora_{model_name.replace('/', '_')}_{ts}_correctness.json")
        with open(json_path, "w") as jf:
            json.dump(correctness, jf, indent=2)

        csv_path = os.path.join(base_dir, f"lora_{model_name.replace('/', '_')}_{ts}_token_count_stats.csv")
        with open(csv_path, "w", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow(["track","chain_idx","step","token_count","correct","tags","prediction","correct_answer","reasoning"])
            for track, records in self.token_stats.items():
                for rec in records:
                    reasoning_text = str(rec.get("reasoning", "")).replace("\n", " ").replace('"', '""')
                    writer.writerow([
                        track,
                        rec["chain_idx"],
                        rec["step"],
                        rec["token_count"],
                        rec["correct"],
                        ";".join(rec.get("tags", [])),
                        rec.get("prediction", ""),
                        rec.get("correct_answer", ""),
                        reasoning_text
                    ])

        return json_path, csv_path

    def _save_single_task_json(self, track: str, chain_idx: int, detailed_predictions: List[Dict[str, Any]]) -> None:
        """Save detailed JSON file for a single task"""
        if not self.detailed_output_dir:
            return

        track_dir = os.path.join(self.detailed_output_dir, track)
        original_filename = self.dataset.filenames[chain_idx]
        task_filename = f"{track}_{original_filename}.json"
        task_filepath = os.path.join(track_dir, task_filename)

        include_reasoning = not "_no_reasoning" in track
        include_correction = not "_no_correction" in track

        task_data = {
            "metadata": {
                "base_model": self.base_model,
                "lora_model": self.lora_model,
                "task_path": track,
                "chain_index": chain_idx,
                "include_reasoning": include_reasoning,
                "include_correction": include_correction,
                "total_steps": len(detailed_predictions),
                "final_accuracy": sum(p["correct"] for p in detailed_predictions[1:]) / max(1, len(detailed_predictions) - 1) if len(detailed_predictions) > 1 else 0,
            },
            "predictions": detailed_predictions
        }

        with open(task_filepath, 'w') as f:
            json.dump(task_data, f, indent=2)

    def evaluate(self):
        """Run evaluation on all tracks"""
        tasks = []
        for path in self.tracks:
            if path.startswith("explicit"):
                chains = self.dataset.explicit_data
            elif path.startswith("implicit_shuffled"):
                chains = self.dataset.implicit_shuffled_data
            else:
                chains = self.dataset.implicit_data

            for idx, chain in enumerate(chains):
                # Check if already exists
                if self.detailed_output_dir:
                    track_dir = os.path.join(self.detailed_output_dir, path)
                    original_filename = self.dataset.filenames[idx]
                    task_filename = f"{path}_{original_filename}.json"
                    task_filepath = os.path.join(track_dir, task_filename)

                    if os.path.exists(task_filepath):
                        print(f"Skipping {path}/{original_filename} - already exists")
                        continue

                tasks.append((path, idx, chain, self.base_model, self.lora_model, self.use_4bit))

        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = {executor.submit(_evaluate_batch, t): t for t in tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
                path, idx, scores, length, step_records, detailed_predictions = future.result()
                self.tally_score_per_prompt[path][idx] = scores
                self.length_score_per_prompt[path][idx] = length
                self.token_stats[path].extend(step_records)

                # Save detailed JSON
                if self.detailed_output_dir:
                    self._save_single_task_json(path, idx, detailed_predictions)


def main():
    """Main evaluation function"""
    parser = ArgumentParser(description="Evaluate LoRA finetuned models on logical reasoning tasks")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing JSON example files")
    parser.add_argument("--base-model", type=str, required=True, help="Base model name")
    parser.add_argument("--lora-model", type=str, required=True, help="Path to LoRA weights")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--results-dir", type=str, default="lora_results", help="Directory for metrics")
    parser.add_argument("--detailed-output-dir", type=str, default=None, help="Directory for detailed task JSONs")
    parser.add_argument("--enable-truncated", action="store_true", help="Use truncated reasoning")
    parser.add_argument("--enable-shuffled", action="store_true", help="Use shuffled datasets")
    parser.add_argument("--use-4bit", action="store_true", default=True, help="Use 4-bit quantization")
    parser.add_argument("--use-8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--no-quantization", action="store_true", help="Disable quantization")

    args = parser.parse_args()

    # Handle quantization
    use_4bit = args.use_4bit and not args.no_quantization and not args.use_8bit
    use_8bit = args.use_8bit and not args.no_quantization

    # Load dataset
    dataset = LogicDataset()
    dataset.read_dir(args.data_dir, args.enable_truncated)

    print(f"\n{'='*80}")
    print(f"LoRA Model Evaluation")
    print(f"{'='*80}")
    print(f"Base Model: {args.base_model}")
    print(f"LoRA Model: {args.lora_model}")
    print(f"Dataset: {args.data_dir}")
    print(f"Examples: {len(dataset)}")
    print(f"Quantization: {'4-bit' if use_4bit else '8-bit' if use_8bit else 'None'}")
    print(f"{'='*80}\n")

    # Create evaluator
    evaluator = LoRAEvaluator(
        dataset=dataset,
        base_model=args.base_model,
        lora_model=args.lora_model,
        batch_size=args.batch_size,
        use_4bit=use_4bit,
        shuffled=args.enable_shuffled,
        detailed_output_dir=args.detailed_output_dir
    )

    # Run evaluation
    evaluator.evaluate()

    # Save metrics
    os.makedirs(args.results_dir, exist_ok=True)
    model_name = Path(args.lora_model).name
    json_path, csv_path = evaluator.save_metrics(model_name=model_name, base_dir=args.results_dir)

    print(f"\n{'='*80}")
    print(f"Evaluation Complete!")
    print(f"{'='*80}")
    print(f"Metrics written to:")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    if args.detailed_output_dir:
        print(f"Detailed task JSONs in: {args.detailed_output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
