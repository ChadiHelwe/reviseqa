"""Phase 0.5 §1.3 — Timeout-Uncertain audit.

For every state in verified-400 whose gold answer is Uncertain (initial or
post-edit), verify that U is established by *completed* two-direction
unprovability — Prover9 exhaustion and/or a Mace4 countermodel — rather than
a timeout. The pipeline logs record no prover budget metadata, so all gold-U
states are re-run at generous budget.

Writes tables/timeout_u_audit.csv and prints the TIMEOUT_AS_U rate.
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DATA = os.path.join(REPO, "reviseqa_data", "nl", "verified-400")
OUT = os.path.join(REPO, "analysis", "phase0_5")

from ladr import fol_to_ladr, run_mace4, run_prover9  # noqa: E402

P9_BUDGET = 600
M4_BUDGET = 60


def gold_u_states():
    """(example_id, turn, theory_fol, conclusion_fol); turn 0 = initial."""
    states = []
    for fname in sorted(os.listdir(DATA)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(DATA, fname)) as f:
            d = json.load(f)
        if d["answer"] == "Uncertain":
            states.append((fname[:-5], 0, d["original_context_fol"], d["conclusion_fol"]))
        for i, e in enumerate(d["edits"], 1):
            if e["answer"] == "Uncertain":
                states.append((fname[:-5], i, e["edited_context_fol"], e["conclusion_fol"]))
    return states


def main():
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    states = gold_u_states()
    print(f"gold-U states in verified-400: {len(states)} "
          f"(a structural finding in itself — U is nearly absent from v1 gold)")
    rows = []
    for ex_id, turn, ctx, conc in states:
        theory = [fol_to_ladr(s) for s in ctx]
        q = fol_to_ladr(conc)
        q_p9, _ = run_prover9(theory, q, P9_BUDGET)
        negq_p9, _ = run_prover9(theory, f"-({q})", P9_BUDGET)
        q_cm, _ = run_mace4(theory, q, M4_BUDGET)
        negq_cm, _ = run_mace4(theory, f"-({q})", M4_BUDGET)
        consist, _ = run_mace4(theory, None, M4_BUDGET)
        q_unprovable_certified = q_p9 == "EXHAUSTED" or q_cm == "MODEL"
        negq_unprovable_certified = negq_p9 == "EXHAUSTED" or negq_cm == "MODEL"
        if q_p9 == "PROVED" or negq_p9 == "PROVED":
            status = "MISLABELED_NOT_U"
        elif q_unprovable_certified and negq_unprovable_certified:
            status = "U_CONFIRMED_COMPLETED"
        else:
            status = "TIMEOUT_AS_U"
        rows.append({"example_id": ex_id, "turn": turn,
                     "q_prover9": q_p9, "negq_prover9": negq_p9,
                     "q_countermodel": q_cm, "negq_countermodel": negq_cm,
                     "consistency_mace4": consist, "status": status})
        print(f"  {ex_id} t{turn}: Q={q_p9}/{q_cm} negQ={negq_p9}/{negq_cm} -> {status}")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "tables", "timeout_u_audit.csv"), index=False)
    n_bad = int((out.status == "TIMEOUT_AS_U").sum()) if len(out) else 0
    n_mis = int((out.status == "MISLABELED_NOT_U").sum()) if len(out) else 0
    print(f"\nTIMEOUT_AS_U rate: {n_bad}/{len(out)}; mislabeled: {n_mis}/{len(out)}")


if __name__ == "__main__":
    main()
