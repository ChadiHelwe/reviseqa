import os
import json
import glob
import shutil
import argparse
import csv
import time
from openai import OpenAI
import instructor
from instructor import Mode
from pydantic import BaseModel
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


class AnswerEnum(str, Enum):
    TRUE = "True"
    FALSE = "False"


class StructuredResponse(BaseModel):
    reasoning: str
    mistake: str
    answer: AnswerEnum


def verify_file_model(filepath, client, model_name, fol_dir=None):
    with open(filepath, "r") as f:
        data = json.load(f)
    edits = data.get("edits", [])
    fol_data = None
    if fol_dir:
        fol_path = os.path.join(fol_dir, os.path.basename(filepath))
        with open(fol_path, "r") as ff:
            fol_data = json.load(ff)
    results = []
    cache_hit = True
    with open(filepath, "r") as cf:
        cached_data = json.load(cf)
    edits = cached_data.get("edits", [])
    for edit in edits:
        mr = edit.get("model_results", {}).get(model_name)
        if mr and len(mr) > 0:
            entry = mr[0]
            results.append((entry.get("verified"), entry.get("mistake")))
        else:
            cache_hit = False

    #         break
    if cache_hit:
        file_verified = all(v for v, _ in results)
        return results, file_verified
    results = []
    subject = fol_data.get("subject_name") if fol_data else None
    category = fol_data.get("subject_category") if fol_data else None
    facts = fol_data.get("context_facts", []) if fol_data else []
    for edit in edits:
        fol_list = edit.get("edited_context_fol", [])
        nl_list = edit.get("edited_natural_language_context", [])
        if len(fol_list) != len(nl_list):
            results.append((False, "length mismatch"))
            continue
        prompt_lines = []
        if subject and category:
            prompt_lines.append(f"Subject: {subject} (Category: {category})")
        if facts:
            prompt_lines.append("Context facts:")
            for fact in facts:
                prompt_lines.append(f"- Text: {fact['text']} FOL: {fact['fol']}")
        prompt_lines.append(
            "For the following pair, does the FOL statement correctly correspond to the natural language statement? Reason first, point out the mistake afterwards (can be none), and finally answer True or False."
            "Do not be overly strict, have some kind of understanding. Your point is to look out for incorrect mappings, only. Think carefully."
        )
        for fol, nl in zip(fol_list, nl_list):
            prompt_lines.append(f"FOL: {fol}")
            prompt_lines.append(f"NL: {nl}")
        prompt = "\n".join(prompt_lines)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an assistant that reasons before only answering True or False, "
                    "identifying the mistake correctly so we can reflect on it later. "
                    "Verify if the FOL statement matches the natural language."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        response: StructuredResponse = client.chat.completions.create(
            model=model_name,
            messages=messages,
            response_model=StructuredResponse,
            extra_body={"provider": {"require_parameters": True}},
        )
        verified = response.answer == AnswerEnum.TRUE
        results.append((verified, response.mistake))
    file_verified = all(v for v, _ in results)
    return results, file_verified


def process_file(
    filepath, client, model_names, fol_dir=None, max_retries=3, retry_delay=2
):
    if fol_dir:
        fol_path = os.path.join(fol_dir, os.path.basename(filepath))
        if not os.path.exists(fol_path):
            return filepath, None, None
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        total_entries = len(data.get("edits", []))
    except Exception:
        return filepath, None, None

    votes = {}
    results_per_model = {}
    cached_file = os.path.join(
        os.path.dirname(filepath), "cached_votes", os.path.basename(filepath)
    )
    shutil.copy(filepath, cached_file)
    for model in model_names:
        results_list = None
        file_pass = None
        for attempt in range(1, max_retries + 1):
            try:
                results_list, file_pass = verify_file_model(
                    cached_file, client, model, fol_dir
                )
                break
            except Exception:
                if attempt < max_retries:
                    time.sleep(retry_delay)
        if results_list is None:
            return filepath, None, None
        votes[model] = file_pass
        results_per_model[model] = results_list

    with open(cached_file, "r+") as f:
        data = json.load(f)
        for idx, edit in enumerate(data.get("edits", [])):
            edit.setdefault("model_results", {})
            for model in model_names:
                verified, mistake = results_per_model[model][idx]
                edit["model_results"][model] = [
                    {"verified": verified, "mistake": mistake}
                ]
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()

    return cached_file, votes, total_entries


def main():
    parser = argparse.ArgumentParser(
        description="Verify JSON edits with FOL and NL context."
    )
    parser.add_argument(
        "--input_dir", required=True, help="Directory containing JSON files to verify."
    )
    parser.add_argument(
        "--fol_dir",
        default=None,
        help="Optional directory containing FOL JSONs with predicates and context facts.",
    )
    parser.add_argument(
        "--model_names",
        nargs="+",
        default=["google/gemini-2.5-flash-preview"],
        help="LLM model identifiers for OpenRouter.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Number of parallel workers."
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable is not set.")
        exit(1)

    openai_client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    guided_client = instructor.from_openai(
        openai_client, mode=Mode.OPENROUTER_STRUCTURED_OUTPUTS
    )

    verified_dir = os.path.join(args.input_dir, "verified")
    unverified_dir = os.path.join(args.input_dir, "unverified")
    skipped_dir = os.path.join(args.input_dir, "skipped_files")
    cached_votes = os.path.join(args.input_dir, "cached_votes")

    for directory in (verified_dir, unverified_dir, skipped_dir):
        if os.path.exists(directory):
            shutil.rmtree(directory)
    os.makedirs(verified_dir, exist_ok=True)
    os.makedirs(unverified_dir, exist_ok=True)
    os.makedirs(skipped_dir, exist_ok=True)
    os.makedirs(cached_votes, exist_ok=True)

    json_files = glob.glob(os.path.join(args.input_dir, "*.json"))
    stats = {}
    stats["total_files_processed"] = len(json_files)
    total_edits_original = 0
    explicit = []
    normal_files = []
    for filepath in json_files:
        with open(filepath, "r") as f:
            data = json.load(f)
        edits = data.get("edits", [])
        total_edits_original += len(edits)
        if edits and edits[0].get("edit_number", 1) > 1:
            explicit.append((filepath, len(edits)))
        else:
            normal_files.append(filepath)
    stats["total_edits_original"] = total_edits_original
    stats["total_files_explicit_dropped"] = len(explicit)
    stats["total_edits_explicit_dropped"] = sum(e for _, e in explicit)
    for filepath, _ in explicit:
        base = os.path.basename(filepath)
        newname = base.replace(".json", "_explicit.json")
        shutil.copy(filepath, os.path.join(unverified_dir, newname))

    results = []
    skipped = []
    with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        future_to_file = {
            executor.submit(
                process_file, filepath, guided_client, args.model_names, args.fol_dir
            ): filepath
            for filepath in normal_files
        }
        for future in tqdm(
            as_completed(future_to_file),
            total=len(future_to_file),
            desc="Processing files",
        ):
            filepath, votes, total_entries = future.result()
            if votes is None:
                skipped.append(filepath)
            else:
                results.append((filepath, votes, total_entries))

    for filepath in skipped:
        shutil.copy(filepath, os.path.join(skipped_dir, os.path.basename(filepath)))
        print(f"Skipped {os.path.basename(filepath)} due to evaluation errors")

    csv_path = os.path.join(args.input_dir, "verification_summary.csv")
    preserved_count = 0
    preserved_edits = 0
    pruned_count = 0
    edits_pruned = 0
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["filename", "total_entries"] + args.model_names + ["majority"]
        writer.writerow(header)
        for filepath, votes, total_entries in results:
            votes_list = [1 if votes[m] else 0 for m in args.model_names]
            majority = sum(votes_list) > 0
            row = [os.path.basename(filepath), total_entries] + votes_list + [majority]
            writer.writerow(row)

            base = os.path.basename(filepath)
            if majority:
                shutil.copy(filepath, os.path.join(verified_dir, base))
                preserved_count += 1
                preserved_edits += total_entries
            else:
                shutil.copy(filepath, os.path.join(unverified_dir, base))
                with open(filepath, "r") as f_in:
                    data_full = json.load(f_in)
                last_good = -1
                for idx, edit in enumerate(data_full.get("edits", [])):
                    ver_counts = sum(
                        1
                        for m in args.model_names
                        if edit["model_results"][m][0]["verified"]
                    )
                    if ver_counts > len(args.model_names) / 2:
                        last_good = idx
                    else:
                        break
                if last_good >= 0:
                    truncated = data_full.copy()
                    truncated["edits"] = data_full["edits"][: last_good + 1]
                    trunc_name = base.replace(".json", "_truncated.json")
                    trunc_path = os.path.join(verified_dir, trunc_name)
                    with open(trunc_path, "w") as f_out:
                        json.dump(truncated, f_out, indent=2)
                    pruned_count += 1
                    edits_pruned += len(data_full["edits"]) - (last_good + 1)

    stats["total_files_preserved"] = preserved_count
    stats["total_files_pruned"] = pruned_count
    stats["total_files_unverified"] = len(results) - preserved_count
    stats["total_files_skipped"] = len(skipped)
    stats["total_edits_preserved"] = preserved_edits
    stats["total_edits_pruned"] = edits_pruned

    print(f"Summary CSV written to {csv_path}")
    stats_path = os.path.join(args.input_dir, "dataset_statistics.json")
    with open(stats_path, "w") as sf:
        json.dump(stats, sf, indent=2)


if __name__ == "__main__":
    main()
