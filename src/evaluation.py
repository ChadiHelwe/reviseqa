import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from argparse import ArgumentParser
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from types import SimpleNamespace
import random
from typing import List, Dict, Any
import re

import instructor
import tiktoken
from tqdm import tqdm
from instructor import Mode
from openai import OpenAI
from pydantic import BaseModel
import functools
import requests
from dotenv import load_dotenv
from confidence import lor, lo



load_dotenv()


def extract_answer_from_prediction(prediction):
    """
    Extract the answer from a prediction string that may contain JSON with reasoning and answer.

    Args:
        prediction: String containing the prediction, possibly with JSON structure

    Returns:
        tuple: (extracted_answer, extracted_reasoning)
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

    # Try to find JSON block within the prediction using multiple patterns
    json_patterns = [
        # Look for complete JSON with reasoning and answer
        r'\{[^{}]*"reasoning"[^{}]*"answer"[^{}]*\}',
        # Look for JSON that might span multiple lines
        r'\{.*?"reasoning".*?"answer".*?\}',
        # Look for any JSON-like structure with answer field
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

    # Try plain text format first: "Reasoning: ... Answer: ..."
    extracted_answer = ""
    extracted_reasoning = ""

    # Extract reasoning from plain text format (case-insensitive)
    reasoning_text_patterns = [
        r'Reasoning:\s*(.*?)(?=Answer:|answer:|$)',
        r'reasoning:\s*(.*?)(?=Answer:|answer:|$)',
        r'REASONING:\s*(.*?)(?=ANSWER:|Answer:|answer:|$)',
    ]

    for pattern in reasoning_text_patterns:
        match = re.search(pattern, prediction, re.DOTALL | re.IGNORECASE)
        if match:
            extracted_reasoning = match.group(1).strip()
            break

    # Extract answer from plain text format (case-insensitive)
    answer_text_patterns = [
        r'Answer:\s*(True|False|Uncertain|true|false|uncertain)',
        r'answer:\s*(True|False|Uncertain|true|false|uncertain)',
        r'ANSWER:\s*(True|False|Uncertain|true|false|uncertain)',
        r'Answer:\s*([^\n\r]+)',
        r'answer:\s*([^\n\r]+)',
        r'ANSWER:\s*([^\n\r]+)',
    ]

    for pattern in answer_text_patterns:
        match = re.search(pattern, prediction, re.IGNORECASE)
        if match:
            extracted_answer = match.group(1).strip()
            break

    # If no plain text format found, try JSON field patterns
    if not extracted_answer:
        json_answer_patterns = [
            r'"answer":\s*(true|false|uncertain|True|False|Uncertain)',
            r'"answer":\s*"([^"]*)"',
            r'"answer":\s*([^,}\s]+)',
        ]

        for pattern in json_answer_patterns:
            match = re.search(pattern, prediction, re.IGNORECASE)
            if match:
                extracted_answer = match.group(1)
                break

    if not extracted_reasoning:
        json_reasoning_patterns = [
            r'"reasoning":\s*"([^"]*)"',
            r'"reasoning":\s*"([^"]*?)"(?=\s*,|\s*})',
        ]

        for pattern in json_reasoning_patterns:
            match = re.search(pattern, prediction, re.DOTALL)
            if match:
                extracted_reasoning = match.group(1)
                break

    # Clean up extracted values
    if extracted_answer.lower() in ['true', 'false', 'uncertain']:
        extracted_answer = extracted_answer.capitalize()
    elif extracted_answer.startswith('"') and extracted_answer.endswith('"'):
        extracted_answer = extracted_answer[1:-1]

    # Final fallback: look for standalone boolean words
    if not extracted_answer:
        standalone_match = re.search(r'\b(True|False|Uncertain|true|false|uncertain)\b', prediction, re.IGNORECASE)
        if standalone_match:
            extracted_answer = standalone_match.group(1).capitalize()

    # If still no answer found, return the original prediction as answer
    if not extracted_answer:
        extracted_answer = prediction.strip()

    return extracted_answer, extracted_reasoning


# Following exact format from ProverQA
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

            # original sample
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

                # determine tags
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

                # build explicit context edits breakdown
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

            # shuffled implicit
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
            # Store filename without .json extension
            self.filenames.append(fname[:-5] if fname.endswith('.json') else fname)

    def __len__(self):
        return len(self.explicit_data)

class AnswerEnum(str, Enum):
    TRUE = "True"
    FALSE = "False"
    UNCERTAIN = "Uncertain"

class StructuredResponse(BaseModel):
    reasoning: str
    answer: AnswerEnum

@functools.lru_cache
def model_supports_structured(model_slug: str) -> bool:
    resp = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        timeout=10,
    )
    supported = {
        m["id"]: set(m.get("parameters_supported", []))
        for m in resp.json()["data"]
    }
    return "json_schema" in supported.get(model_slug, ())

class Conversation:
    def __init__(self, model_name: str, guided: bool = True) -> None:
        client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
        if guided:
            self.client = instructor.from_openai(client, mode=Mode.OPENROUTER_STRUCTURED_OUTPUTS)
        else:
            self.client = client
        self.model_name = model_name
        self.guided = guided

    def init_conversation(
        self, context: str, question: str, reasoning: str, answer: str
    ) -> None:
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

    def send_request(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        attempt = 0
        if not self.guided:
            while attempt < 5:
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name, messages=self.messages,
                        extra_body={"provider": {"sort": "throughput"}}
                    )
                    out = response.choices[0].message.content
                    self.messages.append({"role": "assistant", "content": out})
                    return out
                except Exception as e:
                    print(f"Error during API call: {e}. Retrying...")
                    attempt += 1
                    continue
        
            out = json.dumps({"reasoning": "ERROR", "mistake": "ERROR", "answer": "ERROR"})
            self.messages.append({"role": "assistant", "content": out})
            return out



        try:
            if "claude" in self.model_name:
                response: StructuredResponse = self.client.chat.completions.create(
                    extra_body={"provider": {"sort": "throughput"}},
                    model=self.model_name,
                    messages=self.messages,
                    response_model=StructuredResponse,
                    max_retries=2,
                )
                return response
            else:
                response: StructuredResponse = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.messages,
                    response_model=StructuredResponse,
                    extra_body={"provider": {"require_parameters": True, "sort": "throughput"}},
                    max_retries=2,
                )
                return response
        except Exception as e:
            print(f"Error during API call: {e}")
            err_json = json.dumps({"reasoning": "ERROR", "mistake": "ERROR", "answer": "ERROR"})
            self.messages.append({"role": "assistant", "content": err_json})
            return SimpleNamespace(reasoning="ERROR", answer="ERROR")


def _evaluate_batch(args):
    path, start_idx, batch_entries, model_name, guided = args
    batch_scores = []
    token_counts = []
    step_records: List[Dict[str, Any]] = []
    detailed_predictions: List[Dict[str, Any]] = []
    length = 1
    mistake = False
    prev_correct = True
    recovery_count = 0

    if guided and not model_supports_structured(model_name):
        guided = False

    enc = tiktoken.get_encoding("cl100k_base")
    conv = Conversation(model_name=model_name, guided=guided)

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

        # token count
        token_count_history = sum(len(enc.encode(msg["content"])) for msg in conv.messages)
        token_counts.append(token_count_history)
        response = conv.send_request(role="user", content=prompt)
        # Extract prediction, reasoning, and correctness
        if hasattr(response, "answer"):
            predicted = (
                response.answer.value
                if hasattr(response.answer, "value")
                else response.answer
            )
            reasoning_output = getattr(response, "reasoning", "")
            correct_flag = int(predicted == entry.answer)
        else:
            try:
                parsed = json.loads(response)
                predicted = parsed.get("answer", "")
                reasoning_output = parsed.get("reasoning", "")
                correct_flag = int(entry.answer == predicted)
            except (json.JSONDecodeError, TypeError):
                # If response is not valid JSON, treat the whole response as prediction
                predicted = response if isinstance(response, str) else str(response)
                reasoning_output = ""

                # Try to extract reasoning from the response string if it contains JSON-like structure
                if isinstance(response, str) and '"reasoning"' in response and '"answer"' in response:
                    try:
                        # Try to find and parse JSON within the response using better regex
                        import re
                        # Look for JSON structure with proper nesting support
                        json_match = re.search(r'\{.*?"reasoning".*?"answer".*?\}', response, re.DOTALL)
                        if json_match:
                            json_str = json_match.group()
                            inner_parsed = json.loads(json_str)
                            predicted = inner_parsed.get("answer", predicted)
                            reasoning_output = inner_parsed.get("reasoning", "")
                        else:
                            # Fallback: try to parse the entire response as JSON
                            inner_parsed = json.loads(response.strip())
                            predicted = inner_parsed.get("answer", predicted)
                            reasoning_output = inner_parsed.get("reasoning", "")
                    except:
                        # If all JSON parsing fails, try to extract answer from string patterns
                        answer_match = re.search(r'"answer":\s*"([^"]*)"', response)
                        reasoning_match = re.search(r'"reasoning":\s*"([^"]*)"', response, re.DOTALL)
                        if answer_match:
                            predicted = answer_match.group(1)
                        if reasoning_match:
                            reasoning_output = reasoning_match.group(1)

                correct_flag = int(entry.answer == predicted)

        # Additional layer: try to extract answer from complex JSON predictions
        if correct_flag == 0 and isinstance(response, str):
            final_predicted, final_reasoning = extract_answer_from_prediction(response)
            if final_predicted != predicted:  # Only update if we got a different result
                predicted = final_predicted
                if final_reasoning:  # Update reasoning if we extracted it
                    reasoning_output = final_reasoning
                correct_flag = int(entry.answer == predicted)

        batch_scores.append(correct_flag)

        # Record detailed prediction data
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

        # record step with tags (for compatibility) and additional prediction data
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

class Evaluator:
    def __init__(
        self,
        dataset: LogicDataset,
        batch_size: int = 1,
        model_name: str = "google/gemini-2.5-flash-preview",
        guided: bool = True,
        shuffled: bool = False,
        detailed_output_dir: str = None
    ) -> None:
        self.guided = guided
        self.dataset = dataset
        self.model_name = model_name
        self.batch_size = batch_size
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

        # Create detailed output directory structure if specified
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

    def return_step_by_step_performance(self) -> Dict[str, Dict[str, int]]:
        """Calculate performance at each step position across all examples"""
        step_performance: Dict[str, Dict[str, int]] = {}

        for track in self.tracks:
            step_counts = {}
            total_examples = 0

            # Count correct answers at each step position
            for rec in self.token_stats[track]:
                step = rec["step"]
                if step not in step_counts:
                    step_counts[step] = {"correct": 0, "total": 0}

                step_counts[step]["total"] += 1
                step_counts[step]["correct"] += rec["correct"]

            # Get total number of examples for this track
            if track.startswith("explicit"):
                total_examples = len(self.dataset.explicit_data)
            elif track.startswith("implicit_shuffled"):
                total_examples = len(self.dataset.implicit_shuffled_data)
            else:
                total_examples = len(self.dataset.implicit_data)

            # Format the results
            track_performance = {}
            for step in sorted(step_counts.keys()):
                track_performance[str(step)] = step_counts[step]["correct"]
            track_performance["total_examples"] = total_examples

            step_performance[track] = track_performance

        return step_performance

    def save_metrics(self, model_name: str, base_dir: str = "results") -> tuple[str, str]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        tally = self.return_tally_sum()
        by_diff = self.return_length_by_difficulty()
        step_performance = self.return_step_by_step_performance()

        # Degradation buckets with totals & correct
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

        # Permutation / edit-type stats
        perm_stats: Dict[str, Dict[str, Dict[str, int]]] = {t: {} for t in self.tracks}
        for track in self.tracks:
            for rec in self.token_stats[track]:
                for tag in rec.get("tags", ["untagged"]):
                    ps = perm_stats[track].setdefault(tag, {"total":0, "correct":0})
                    ps["total"] += 1
                    ps["correct"] += rec["correct"]

        alpha = 0.05
        ci = {}
        for track, records in self.token_stats.items():
            total = len(records)
            success = sum(rec["correct"] for rec in records)
            lo_rand = lor(success, total, alpha/2)[0]
            hi_rand = 1 - lor(total - success, total, alpha/2)[0]
            lo_det, hi_det = lo(success, total, alpha=alpha)
            ci[track] = {
                "p": success / total,
                "rand_lower": float(lo_rand),
                "rand_upper": float(hi_rand),
                "det_lower": float(lo_det),
                "det_upper": float(hi_det),
            }

        correctness = {
            "timestamp": datetime.now().isoformat(),
            "metadata": {"model_name": self.model_name, "dataset_length": len(self.dataset), "batch_size": self.batch_size},
            "total_per_track": {t: len(self.token_stats[t]) for t in self.tracks},
            "tally_sum": tally,
            "length_by_difficulty": by_diff,
            "degradation_buckets": degradation_buckets,
            "permutation_stats": perm_stats,
            # "confidence_intervals_95": ci,
        }

        json_path = os.path.join(base_dir, f"{model_name.replace('/', '_')}_{ts}_correctness.json")
        with open(json_path, "w") as jf:
            json.dump(correctness, jf, indent=2)

        csv_path = os.path.join(base_dir, f"{model_name.replace('/', '_')}_{ts}_token_count_stats.csv")
        with open(csv_path, "w", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow(["track","chain_idx","step","token_count","correct","tags","prediction","correct_answer","reasoning"])
            for track, records in self.token_stats.items():
                for rec in records:
                    # Clean reasoning text for CSV (replace newlines and quotes)
                    reasoning = rec.get("reasoning", "")
                    if isinstance(reasoning, list):
                        reasoning_text = " ".join(str(item) for item in reasoning)
                    else:
                        reasoning_text = str(reasoning)
                    reasoning_text = reasoning_text.replace("\n", " ").replace('"', '""')
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
        """Save detailed JSON file for a single task immediately after completion"""
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
                "model_name": self.model_name,
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
        tasks = []
        for path in self.tracks:
            if path.startswith("explicit"):
                chains = self.dataset.explicit_data
            elif path.startswith("implicit_shuffled"):
                chains = self.dataset.implicit_shuffled_data
            else:
                chains = self.dataset.implicit_data
            for idx, chain in enumerate(chains):
                # Check if detailed result file already exists
                if self.detailed_output_dir:
                    track_dir = os.path.join(self.detailed_output_dir, path)
                    original_filename = self.dataset.filenames[idx]
                    task_filename = f"{path}_{original_filename}.json"
                    task_filepath = os.path.join(track_dir, task_filename)

                    if os.path.exists(task_filepath):
                        print(f"Skipping {path}/{original_filename} - result already exists")
                        continue

                tasks.append((path, idx, chain, self.model_name, self.guided))

        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = {executor.submit(_evaluate_batch, t): t for t in tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
                path, idx, scores, length, step_records, detailed_predictions = future.result()
                self.tally_score_per_prompt[path][idx] = scores
                self.length_score_per_prompt[path][idx] = length
                self.token_stats[path].extend(step_records)

                # Save detailed JSON file immediately after task completion
                if self.detailed_output_dir:
                    self._save_single_task_json(path, idx, detailed_predictions)

def main():
    """Main function for evaluation"""
    parser = ArgumentParser(description="Evaluate a logic QA dataset with structured or free-form LLM outputs")
    parser.add_argument("--data-dir", type=str, default="src/data", help="Directory containing JSON example files")
    parser.add_argument("--batch-size", type=int, default=32, help="Number of parallel worker processes")
    parser.add_argument("--model-name", type=str, default="google/gemini-2.5-flash-preview", help="LLM model identifier")
    parser.add_argument("--guided", action="store_true", help="Enable structured-output guided mode")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory in which to save metrics")
    parser.add_argument("--detailed-output-dir", type=str, default=None, help="Directory in which to save detailed JSON files for each task")
    parser.add_argument("--enable_truncated", action="store_true", help="Use truncated reasoning for evaluation")
    parser.add_argument("--enable_shuffled", action="store_true", help="Use shuffled datasets for evaluation")
    args = parser.parse_args()

    dataset = LogicDataset()
    dataset.read_dir(args.data_dir, args.enable_truncated)

    evaluator = Evaluator(dataset, batch_size=args.batch_size, model_name=args.model_name, guided=args.guided, shuffled=args.enable_shuffled, detailed_output_dir=args.detailed_output_dir)
    evaluator.evaluate()
    os.makedirs(args.results_dir, exist_ok=True)
    json_path, csv_path = evaluator.save_metrics(model_name=args.model_name, base_dir=args.results_dir)
    print(f"Metrics written to:\n  {json_path}\n  {csv_path}")
    if args.detailed_output_dir:
        print(f"Detailed task JSON files written to: {args.detailed_output_dir}")


if __name__ == "__main__":
    main()