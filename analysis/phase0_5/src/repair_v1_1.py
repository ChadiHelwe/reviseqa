"""Phase 0.5 §1.4 — v1.1 repair script.

For all 930 verification-preserved examples (majority=True in
reviseqa_data/nl/verification_summary.csv):

(b) `edits_made` regenerated from the actual FOL context diffs, with NL text
    recovered via the parallel NL/FOL context arrays and fact/rule classified
    by FOL syntax. The original field is preserved as `edits_made_v1`.
(c) Mace4 consistency stamp on every turn state (initial + each edit).
    Any example with an inconsistent or malformed state is QUARANTINED.
(d) Prover-budget metadata per direction per turn: proved / disproved /
    exhausted / timeout, plus Mace4 countermodel certificates, the re-derived
    label, and its agreement with the stored answer.

(a) — the demo-answer harness fix — is applied directly to
`src/evaluation.py` (see CHANGELOG).

Raw dataset files are read-only; v1.1 is written under
analysis/phase0_5/v1.1/ (nl/ + quarantine/ + CHANGELOG.md + manifest).

Deterministic; parallel over examples (results independent per example).
"""
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
NL = os.path.join(REPO, "reviseqa_data", "nl")
OUT = os.path.join(REPO, "analysis", "phase0_5")
V11 = os.path.join(OUT, "v1.1")

from ladr import fol_to_ladr, run_mace4, run_prover9  # noqa: E402

P9_BUDGET = 60
M4_BUDGET = 30
VERSION = "1.1.0"


def fol_kind(s: str) -> str:
    import re
    return "rule" if re.search(r"[→↔⊕∨∧∀∃]", s) else "fact"


def state_metadata(theory_fol, conclusion_fol):
    theory = [fol_to_ladr(s) for s in theory_fol]
    q = fol_to_ladr(conclusion_fol)
    consist, _ = run_mace4(theory, None, M4_BUDGET)
    contra, _ = run_prover9(theory, None, P9_BUDGET)
    q_p9, _ = run_prover9(theory, q, P9_BUDGET)
    negq_p9, _ = run_prover9(theory, f"-({q})", P9_BUDGET)
    q_cm, _ = run_mace4(theory, q, M4_BUDGET)
    negq_cm, _ = run_mace4(theory, f"-({q})", M4_BUDGET)
    if contra == "PROVED":
        label = "INCONSISTENT"
    elif "ERROR" in (contra, q_p9, negq_p9):
        label = "MALFORMED"
    elif q_p9 == "PROVED":
        label = "True"
    elif negq_p9 == "PROVED":
        label = "False"
    elif (q_p9 == "EXHAUSTED" or q_cm == "MODEL") and \
         (negq_p9 == "EXHAUSTED" or negq_cm == "MODEL"):
        label = "Uncertain"
    else:
        label = "UNRESOLVED"
    return {
        "consistency_mace4": consist,
        "contradiction_prover9": contra,
        "q_prover9": q_p9, "negq_prover9": negq_p9,
        "q_countermodel_mace4": q_cm, "negq_countermodel_mace4": negq_cm,
        "prover9_max_seconds": P9_BUDGET, "mace4_max_seconds": M4_BUDGET,
        "rederived_label": label,
    }


def regen_edits_made(prev_nl, prev_fol, cur_nl, cur_fol):
    prev_map = dict(zip(prev_fol, prev_nl))
    cur_map = dict(zip(cur_fol, cur_nl))
    added = [f for f in cur_fol if f not in set(prev_fol)]
    removed = [f for f in prev_fol if f not in set(cur_fol)]
    em = {"removed_facts": [], "removed_rules": [], "added_facts": [], "added_rules": []}
    for f in removed:
        em[f"removed_{fol_kind(f)}s"].append({"fol": f, "nl": prev_map[f]})
    for f in added:
        em[f"added_{fol_kind(f)}s"].append({"fol": f, "nl": cur_map[f]})
    return em


def repair_example(fname):
    with open(os.path.join(NL, fname)) as f:
        d = json.load(f)
    d["reviseqa_version"] = VERSION
    problems = []

    init_meta = state_metadata(d["original_context_fol"], d["conclusion_fol"])
    init_meta["label_agrees_with_stored"] = init_meta["rederived_label"] == d["answer"]
    d["initial_state_metadata"] = init_meta
    if init_meta["rederived_label"] in ("INCONSISTENT", "MALFORMED"):
        problems.append(("initial", init_meta["rederived_label"]))
    elif not init_meta["label_agrees_with_stored"]:
        problems.append(("initial", f"label_mismatch:{init_meta['rederived_label']}"))

    prev_nl = d["original_context"]
    prev_fol = d["original_context_fol"]
    for i, e in enumerate(d.get("edits", []), 1):
        cur_nl = e["edited_natural_language_context"]
        cur_fol = e["edited_context_fol"]
        e["edits_made_v1"] = e["edits_made"]
        e["edits_made"] = regen_edits_made(prev_nl, prev_fol, cur_nl, cur_fol)
        meta = state_metadata(cur_fol, e["conclusion_fol"])
        meta["label_agrees_with_stored"] = meta["rederived_label"] == e["answer"]
        e["prover_metadata"] = meta
        if meta["rederived_label"] in ("INCONSISTENT", "MALFORMED"):
            problems.append((f"edit_{i}", meta["rederived_label"]))
        elif not meta["label_agrees_with_stored"]:
            problems.append((f"edit_{i}", f"label_mismatch:{meta['rederived_label']}"))
        prev_nl, prev_fol = cur_nl, cur_fol

    quarantined = any(p[1] in ("INCONSISTENT", "MALFORMED") for p in problems)
    return fname, d, problems, quarantined


def main():
    os.makedirs(os.path.join(V11, "nl"), exist_ok=True)
    os.makedirs(os.path.join(V11, "quarantine"), exist_ok=True)
    preserved = [r["filename"] for r in
                 csv.DictReader(open(os.path.join(NL, "verification_summary.csv")))
                 if r["majority"] == "True"]
    print(f"repairing {len(preserved)} preserved examples")

    manifest = []
    n_done = 0
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(repair_example, f): f for f in preserved}
        for fut in as_completed(futures):
            fname, d, problems, quarantined = fut.result()
            dest = "quarantine" if quarantined else "nl"
            with open(os.path.join(V11, dest, fname), "w") as f:
                json.dump(d, f, indent=2)
            manifest.append({
                "filename": fname, "quarantined": quarantined,
                "n_problems": len(problems),
                "problems": ";".join(f"{a}={b}" for a, b in problems),
            })
            n_done += 1
            if n_done % 100 == 0:
                print(f"  {n_done}/{len(preserved)}")

    manifest.sort(key=lambda r: r["filename"])
    import pandas as pd
    mdf = pd.DataFrame(manifest)
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    mdf.to_csv(os.path.join(V11, "manifest.csv"), index=False)

    n_quar = int(mdf.quarantined.sum())
    n_mismatch = int(((~mdf.quarantined) & (mdf.n_problems > 0)).sum())
    n_clean = int((mdf.n_problems == 0).sum())
    print(f"\nclean: {n_clean}, label-mismatch (shipped, flagged): {n_mismatch}, "
          f"quarantined (inconsistent/malformed): {n_quar}")

    changelog = f"""# ReviseQA v1.1 — Changelog

Produced by `analysis/phase0_5/src/repair_v1_1.py` (SPEC_ADDENDUM_A §1.4)
from the 930 verification-preserved v1 examples. Prover9/Mace4 (LADR
2009-11A, Homebrew). Budgets: prover9 {P9_BUDGET}s/direction, mace4
{M4_BUDGET}s, domains ≤ 20.

## Changes vs v1

(a) **Harness demo-answer fix** — `src/evaluation.py` now binds the turn-0
    demonstration answer to the original context's answer (was:
    `edits[-1]["answer"]`, wrong for 207/400 eval examples).
(b) **`edits_made` regenerated** from actual FOL context diffs (NL recovered
    via the parallel NL/FOL arrays; fact/rule classified by FOL syntax).
    The original v1 field is preserved as `edits_made_v1`.
(c) **Mace4 consistency stamp** on every turn state
    (`consistency_mace4` / `contradiction_prover9`). Examples with any
    INCONSISTENT or MALFORMED state are quarantined, not shipped.
(d) **Prover-budget metadata** per direction per turn
    (`q_prover9`, `negq_prover9` ∈ PROVED/EXHAUSTED/TIMEOUT, plus Mace4
    countermodel certificates and budgets), a `rederived_label`, and
    `label_agrees_with_stored`. Stored `answer` fields are unchanged;
    label mismatches are shipped but flagged in `manifest.csv` — exclude
    them when sampling for Study 2.

## Counts

- clean examples: {n_clean}
- shipped with label mismatches (flagged): {n_mismatch}
- quarantined (inconsistent or malformed states): {n_quar}
"""
    with open(os.path.join(V11, "CHANGELOG.md"), "w") as f:
        f.write(changelog)
    print("v1.1 frozen at", V11)


if __name__ == "__main__":
    main()
