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

import instructor
import tiktoken
from tqdm import tqdm
from instructor import Mode
from openai import OpenAI
from pydantic import BaseModel
import functools
import requests

from confidence import lor, lo

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
                        model=self.model_name, messages=self.messages
                    )
                    out = response.choices[0].message.content
                    self.messages.append({"role": "assistant", "content": out})
                    return out
                except:
                    attempt += 1
                    continue
        
            out = json.dumps({"reasoning": "ERROR", "mistake": "ERROR", "answer": "ERROR"})
            self.messages.append({"role": "assistant", "content": out})
            return out



        try:
            if "claude" in self.model_name:
                response: StructuredResponse = self.client.chat.completions.create(
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
                    extra_body={"provider": {"require_parameters": True}},
                    max_retries=2,
                )
                return response
        except Exception:
            err_json = json.dumps({"reasoning": "ERROR", "mistake": "ERROR", "answer": "ERROR"})
            self.messages.append({"role": "assistant", "content": err_json})
            return SimpleNamespace(reasoning="ERROR", answer="ERROR")


def _evaluate_batch(args):
    path, start_idx, batch_entries, model_name, guided = args
    batch_scores = []
    token_counts = []
    step_records: List[Dict[str, Any]] = []
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
        # correctness
        if hasattr(response, "answer"):
            predicted = (
                response.answer.value
                if hasattr(response.answer, "value")
                else response.answer
            )
            correct_flag = int(predicted == entry.answer)
        else:
            try:
                parsed = json.loads(response)
                predicted = parsed.get("answer", "")
                correct_flag = int(entry.answer == predicted)
            except (json.JSONDecodeError, TypeError):
                predicted = response if isinstance(response, str) else str(response)
                correct_flag = int(entry.answer in predicted)


        batch_scores.append(correct_flag)
        # record step with tags
        step_records.append({
            "chain_idx": start_idx,
            "step": step_idx,
            "token_count": token_count_history,
            "correct": correct_flag,
            "tags": entry.tags,
        })

        if correct_flag and not prev_correct:
            recovery_count += 1
        if correct_flag and not mistake:
            length += 1
            prev_correct = True
        elif not correct_flag:
            prev_correct = False
            mistake = True

    return path, start_idx, batch_scores, length, step_records

class Evaluator:
    def __init__(
        self,
        dataset: LogicDataset,
        batch_size: int = 1,
        model_name: str = "google/gemini-2.5-flash-preview",
        guided: bool = True,
        shuffled: bool = False
    ) -> None:
        self.guided = guided
        self.dataset = dataset
        self.model_name = model_name
        self.batch_size = batch_size

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

    def save_metrics(self, model_name: str, base_dir: str = "results") -> tuple[str, str]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        tally = self.return_tally_sum()
        by_diff = self.return_length_by_difficulty()

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
            "confidence_intervals_95": ci,
        }

        json_path = os.path.join(base_dir, f"{model_name.replace('/', '_')}_{ts}_correctness.json")
        with open(json_path, "w") as jf:
            json.dump(correctness, jf, indent=2)

        csv_path = os.path.join(base_dir, f"{model_name.replace('/', '_')}_{ts}_token_count_stats.csv")
        with open(csv_path, "w", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow(["track","chain_idx","step","token_count","correct","tags"])
            for track, records in self.token_stats.items():
                for rec in records:
                    writer.writerow([track, rec["chain_idx"], rec["step"], rec["token_count"], rec["correct"], ";".join(rec.get("tags", []))])

        return json_path, csv_path

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
                tasks.append((path, idx, chain, self.model_name, self.guided))

        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = {executor.submit(_evaluate_batch, t): t for t in tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
                path, idx, scores, length, step_records = future.result()
                self.tally_score_per_prompt[path][idx] = scores
                self.length_score_per_prompt[path][idx] = length
                self.token_stats[path].extend(step_records)

def main():
    """Main function for evaluation"""
    parser = ArgumentParser(description="Evaluate a logic QA dataset with structured or free-form LLM outputs")
    parser.add_argument("--data-dir", type=str, default="src/data", help="Directory containing JSON example files")
    parser.add_argument("--batch-size", type=int, default=32, help="Number of parallel worker processes")
    parser.add_argument("--model-name", type=str, default="google/gemini-2.5-flash-preview", help="LLM model identifier")
    parser.add_argument("--guided", action="store_true", help="Enable structured-output guided mode")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory in which to save metrics")
    parser.add_argument("--enable_truncated", action="store_true", help="Use truncated reasoning for evaluation")
    parser.add_argument("--enable_shuffled", action="store_true", help="Use shuffled datasets for evaluation")
    args = parser.parse_args()

    dataset = LogicDataset()
    dataset.read_dir(args.data_dir, args.enable_truncated)

    evaluator = Evaluator(dataset, batch_size=args.batch_size, model_name=args.model_name, guided=args.guided, shuffled=args.enable_shuffled)
    evaluator.evaluate()
    os.makedirs(args.results_dir, exist_ok=True)
    json_path, csv_path = evaluator.save_metrics(model_name=args.model_name, base_dir=args.results_dir)
    print(f"Metrics written to:\n  {json_path}\n  {csv_path}")


if __name__ == "__main__":
    main()