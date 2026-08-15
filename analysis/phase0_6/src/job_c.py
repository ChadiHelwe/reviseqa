"""Phase 0.6 Job C — pre-edit label re-derivation for the 2 comparator turns
that audited CONSISTENT_GOLD_OK in Phase 0.5 §1.1.

Prediction: the pre-edit states are mislabeled, making the "flips" illusory.
600 s/direction with Mace4 countermodel certificates.

Writes tables/c_leftover_comparator.csv.
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(REPO, "analysis", "phase0_6")
DATA = os.path.join(REPO, "reviseqa_data", "nl", "verified-400")
sys.path.insert(0, os.path.join(REPO, "analysis", "phase0_5", "src"))
from ladr import fol_to_ladr, rederive_gold  # noqa: E402


def main():
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    ca = pd.read_csv(os.path.join(REPO, "analysis", "phase0_5", "tables",
                                  "comparator_audit.csv"))
    ok = ca[(ca.cohort == "comparator_add_flip") & (ca.verdict == "CONSISTENT_GOLD_OK")]
    rows = []
    for r in ok.itertuples():
        d = json.load(open(os.path.join(DATA, f"{r.example_id}.json")))
        t = int(r.turn_index)
        if t > 1:
            pre = d["edits"][t - 2]
            pre_ctx, pre_conc = pre["edited_context_fol"], pre["conclusion_fol"]
            pre_stored = pre["answer"]
        else:
            pre_ctx, pre_conc = d["original_context_fol"], d["conclusion_fol"]
            pre_stored = d["answer"]
        theory = [fol_to_ladr(s) for s in pre_ctx]
        res = rederive_gold(theory, fol_to_ladr(pre_conc), p9_budget=600, m4_budget=60)
        rederived = res["rederived_gold"]
        true_transition = (f"{rederived[0] if rederived in ('True','False','Uncertain') else '?'}"
                           f"→{str(r.stored_gold)[0]}")
        rows.append({
            "example_id": r.example_id, "turn_index": t,
            "stored_transition": r.transition,
            "pre_stored_label": pre_stored, "pre_rederived_label": rederived,
            "pre_q_prover9": res["q_prover9"], "pre_negq_prover9": res["negq_prover9"],
            "pre_q_countermodel": res["q_countermodel"],
            "pre_negq_countermodel": res["negq_countermodel"],
            "pre_consistency_mace4": res["consistency_mace4"],
            "pre_mislabeled": rederived != pre_stored,
            "true_transition": true_transition,
            "flip_illusory": rederived != pre_stored,
        })
        print(rows[-1])
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "tables", "c_leftover_comparator.csv"), index=False)
    n_ill = int(out.flip_illusory.sum())
    print(f"\nJob C: {n_ill}/{len(out)} leftover comparator flips are illusory "
          f"(pre-edit state mislabeled)")


if __name__ == "__main__":
    main()
