"""Build the Phase 0 tidy table from raw eval logs + dataset files.

Reads (read-only):
  detailed_models_results/<provider>/<model>/<track>/<track>_<example_id>.json
  reviseqa_data/nl/verified-400/<example_id>.json

Writes:
  analysis/phase0/tidy.parquet, analysis/phase0/tidy.csv
  analysis/phase0/tables/inventory.csv, pairing.csv, parse_quality.csv,
  gold_consistency.csv

One row per (model, prompting, feedback, setting, example_id, turn).
"""
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from code_edits import (  # noqa: E402
    classify_actual, classify_edit, edit_counts, edit_target, normalize_pred,
    transition,
)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
LOGS = os.path.join(REPO, "detailed_models_results")
DATA = os.path.join(REPO, "reviseqa_data", "nl", "verified-400")
OUT = os.path.join(REPO, "analysis", "phase0")

TRACKS = [
    "explicit", "explicit_no_reasoning",
    "explicit_no_correction", "explicit_no_reasoning_no_correction",
    "implicit", "implicit_no_reasoning",
    "implicit_no_correction", "implicit_no_reasoning_no_correction",
]


def track_attrs(track: str):
    setting = "explicit" if track.startswith("explicit") else "implicit"
    prompting = "standard" if "_no_reasoning" in track else "cot"
    feedback = "_no_correction" not in track
    return setting, prompting, feedback


def load_dataset():
    """example_id -> dict with initial answer and per-edit metadata."""
    ds = {}
    for fname in sorted(os.listdir(DATA)):
        if not fname.endswith(".json"):
            continue
        ex_id = fname[:-5]
        with open(os.path.join(DATA, fname)) as f:
            d = json.load(f)
        edits = []
        prev_fol = set(d["original_context_fol"])
        for e in d.get("edits", []):
            em = e["edits_made"]
            n_add_rec, n_rem_rec = edit_counts(em)
            cur_fol = set(e["edited_context_fol"])
            act_added = cur_fol - prev_fol
            act_removed = prev_fol - cur_fol
            rec_added = {x["fol"] for k in ("added_facts", "added_rules") for x in em[k]}
            rec_removed = {x["fol"] for k in ("removed_facts", "removed_rules") for x in em[k]}
            actual = classify_actual(act_added, act_removed)
            edits.append({
                "edit_number": e.get("edit_number"),
                "modification_type": e.get("modification_type"),
                "answer": e["answer"],
                # primary coding: from the ACTUAL FOL context delta
                "n_add": actual["n_add"],
                "n_rem": actual["n_rem"],
                "edit_class": actual["edit_class"],
                "edit_target": actual["edit_target"],
                # spec (pre-registered) coding: from the recorded edits_made
                "n_add_recorded": n_add_rec,
                "n_rem_recorded": n_rem_rec,
                "edit_class_recorded": classify_edit(em),
                "edit_target_recorded": edit_target(em),
                # delta-integrity audit
                "n_unrecorded_removed": len(act_removed - rec_removed),
                "n_unrecorded_added": len(act_added - rec_added),
                "n_phantom_removed": len(rec_removed - act_removed),
                "n_phantom_added": len(rec_added - act_added),
            })
            prev_fol = cur_fol
        ds[ex_id] = {
            "initial_answer": d["answer"],
            "demo_answer_as_shown": (d["edits"][-1]["answer"] if d.get("edits") else d["answer"]),
            "edits": edits,
        }
    return ds


def iter_models():
    for prov in sorted(os.listdir(LOGS)):
        pdir = os.path.join(LOGS, prov)
        if not os.path.isdir(pdir):
            continue
        for model in sorted(os.listdir(pdir)):
            mdir = os.path.join(pdir, model)
            if os.path.isdir(mdir):
                yield prov, model, mdir


def build():
    ds = load_dataset()
    rows = []
    inventory = []
    for prov, model, mdir in iter_models():
        model_key = f"{prov}/{model}"
        for track in TRACKS:
            tdir = os.path.join(mdir, track)
            if not os.path.isdir(tdir):
                continue
            setting, prompting, feedback = track_attrs(track)
            n_files = 0
            pat = re.compile(rf"^{re.escape(track)}_((?:prev_)?ex_\d+)\.json$")
            for fname in sorted(os.listdir(tdir)):
                m = pat.match(fname)
                if not m:
                    continue
                ex_id = m.group(1)
                meta = ds.get(ex_id)
                if meta is None:
                    raise KeyError(f"log example {ex_id} not in dataset dir {DATA}")
                with open(os.path.join(tdir, fname)) as f:
                    log = json.load(f)
                preds = log["predictions"]
                assert preds[0].get("is_demonstration"), f"{fname}: step 0 not demo"
                n_files += 1
                gold_prev = meta["initial_answer"]
                for step, p in enumerate(preds[1:], start=1):
                    edit = meta["edits"][step - 1]
                    gold_t = edit["answer"]
                    if p["correct_answer"] != gold_t:
                        raise ValueError(
                            f"gold mismatch {model_key}/{track}/{ex_id} step {step}: "
                            f"log={p['correct_answer']} dataset={gold_t}")
                    pred = normalize_pred(p["prediction"])
                    parse_ok = pred != "PARSE_FAIL"
                    rows.append({
                        "model": model_key,
                        "prompting": prompting,
                        "feedback": feedback,
                        "setting": setting,
                        "track": track,
                        "example_id": ex_id,
                        "turn_index": step,
                        "pred_raw": str(p["prediction"])[:500],
                        "pred": pred,
                        "gold_t": gold_t,
                        "gold_prev": gold_prev,
                        "correct": bool(parse_ok and pred == gold_t),
                        "log_correct": bool(p["correct"]),
                        "parse_ok": parse_ok,
                        "n_add": edit["n_add"],
                        "n_rem": edit["n_rem"],
                        "edit_class": edit["edit_class"],
                        "edit_target": edit["edit_target"],
                        "n_add_recorded": edit["n_add_recorded"],
                        "n_rem_recorded": edit["n_rem_recorded"],
                        "edit_class_recorded": edit["edit_class_recorded"],
                        "edit_target_recorded": edit["edit_target_recorded"],
                        "delta_mismatch": bool(
                            edit["n_unrecorded_removed"] or edit["n_unrecorded_added"]
                            or edit["n_phantom_removed"] or edit["n_phantom_added"]),
                        "n_unrecorded_removed": edit["n_unrecorded_removed"],
                        "n_unrecorded_added": edit["n_unrecorded_added"],
                        "modification_type": edit["modification_type"],
                        "answer_changed": gold_t != gold_prev,
                        "transition": transition(gold_prev, gold_t),
                        "demo_answer_as_shown": meta["demo_answer_as_shown"],
                        "initial_answer": meta["initial_answer"],
                    })
                    gold_prev = gold_t
            inventory.append({
                "model": model_key, "track": track, "setting": setting,
                "prompting": prompting, "feedback": feedback, "n_examples": n_files,
            })
    df = pd.DataFrame(rows)
    inv = pd.DataFrame(inventory)
    return df, inv


def pairing_table(df):
    """Explicit/implicit example-ID overlap per model x prompting x feedback."""
    out = []
    for (model, prompting, feedback), g in df.groupby(["model", "prompting", "feedback"]):
        e = set(g.loc[g.setting == "explicit", "example_id"])
        i = set(g.loc[g.setting == "implicit", "example_id"])
        out.append({
            "model": model, "prompting": prompting, "feedback": feedback,
            "n_explicit": len(e), "n_implicit": len(i),
            "n_paired": len(e & i),
            "overlap_pct_of_union": round(100 * len(e & i) / max(1, len(e | i)), 2),
        })
    return pd.DataFrame(out)


def parse_quality_table(df):
    pq = (df.groupby(["model", "track"])
            .agg(n_turns=("parse_ok", "size"),
                 n_parse_fail=("parse_ok", lambda s: int((~s).sum())))
            .reset_index())
    pq["parse_fail_pct"] = (100 * pq.n_parse_fail / pq.n_turns).round(2)
    return pq


def main():
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    df, inv = build()

    # cross-check our correctness coding vs the harness's own flag
    gc = (df.assign(agree=df.correct == df.log_correct)
            .groupby(["model", "track"])
            .agg(n=("agree", "size"), n_disagree=("agree", lambda s: int((~s).sum())))
            .reset_index())

    df.to_parquet(os.path.join(OUT, "tidy.parquet"), index=False)
    df.to_csv(os.path.join(OUT, "tidy.csv"), index=False)
    inv.to_csv(os.path.join(OUT, "tables", "inventory.csv"), index=False)
    pairing_table(df).to_csv(os.path.join(OUT, "tables", "pairing.csv"), index=False)
    parse_quality_table(df).to_csv(os.path.join(OUT, "tables", "parse_quality.csv"), index=False)
    gc.to_csv(os.path.join(OUT, "tables", "gold_consistency.csv"), index=False)

    # FOL delta-integrity audit table (unique example-turn level)
    audit = (df[["example_id", "turn_index", "edit_class", "edit_class_recorded",
                 "delta_mismatch", "n_unrecorded_removed", "n_unrecorded_added",
                 "answer_changed", "modification_type"]]
             .drop_duplicates())
    audit.to_csv(os.path.join(OUT, "tables", "fol_delta_audit.csv"), index=False)

    print(f"tidy rows: {len(df)}")
    print(f"models: {df.model.nunique()}, tracks: {df.track.nunique()}, "
          f"examples: {df.example_id.nunique()}")
    print(f"\ndelta_mismatch example-turns: {int(audit.delta_mismatch.sum())} "
          f"/ {len(audit)}")
    print("\nedit_class (actual) x answer_changed cell counts (all runs):")
    print(df.groupby(["edit_class", "answer_changed"]).size())
    print("\nreclassification: recorded -> actual (unique example-turns):")
    print(audit.groupby(["edit_class_recorded", "edit_class"]).size())


if __name__ == "__main__":
    main()
