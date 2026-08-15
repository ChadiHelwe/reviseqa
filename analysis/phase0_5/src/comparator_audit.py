"""Phase 0.5 §1.1 — Comparator consistency audit.

Cohorts:
  comparator_add_flip : the 18 delta-verified add_only × changed turns
                        (the P1 comparator)
  u_to_f              : the single U→F removal turn
  pure_removal        : the 48 recorded pure-removal turns

For each turn: Mace4 consistency, Prover9 contradiction search, and gold
re-derivation (Q / ¬Q) at 10× the pipeline's original budget (nltk default
60 s → 600 s), plus Mace4 countermodel certificates. The audited theory is
the *displayed* context (`edited_context_fol`, converted) — Phase 0.5 also
showed the stored `prover9_input` diverges semantically from the context on
54 edits, so labels must be re-derived from what models actually saw.

Writes tables/comparator_audit.csv and prints the §1.1 decision rule.
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

from ladr import fol_to_ladr, rederive_gold, run_prover9  # noqa: E402

P9_BUDGET = 600   # 10x the pipeline's nltk default of 60 s
M4_BUDGET = 60


def cohorts():
    df = pd.read_parquet(os.path.join(REPO, "analysis", "phase0", "tidy.parquet"))
    u = df[["example_id", "turn_index", "edit_class", "edit_class_recorded",
            "answer_changed", "transition", "delta_mismatch",
            "n_rem_recorded", "n_add_recorded"]].drop_duplicates()
    comp = u[(u.edit_class == "add_only") & u.answer_changed & (~u.delta_mismatch)]
    utf = u[u.transition == "U→F"]
    pure = u[(u.n_rem_recorded > 0) & (u.n_add_recorded == 0)]
    rows = []
    for name, sub in [("comparator_add_flip", comp), ("u_to_f", utf),
                      ("pure_removal", pure)]:
        for r in sub.itertuples():
            rows.append({"cohort": name, "example_id": r.example_id,
                         "turn_index": int(r.turn_index),
                         "transition": r.transition,
                         "answer_changed": bool(r.answer_changed)})
    return pd.DataFrame(rows).drop_duplicates(subset=["cohort", "example_id", "turn_index"])


def theory_and_goal(ex_id, turn):
    with open(os.path.join(DATA, f"{ex_id}.json")) as f:
        d = json.load(f)
    e = d["edits"][turn - 1]
    theory = [fol_to_ladr(s) for s in e["edited_context_fol"]]
    goal_q = fol_to_ladr(e["conclusion_fol"])
    prev_theory = ([fol_to_ladr(s) for s in d["edits"][turn - 2]["edited_context_fol"]]
                   if turn > 1 else [fol_to_ladr(s) for s in d["original_context_fol"]])
    stored = e["answer"]
    return theory, goal_q, prev_theory, stored


def verdict_of(res, stored):
    g = res["rederived_gold"]
    if g == "INCONSISTENT":
        return "INCONSISTENT"
    if g == stored:
        return "CONSISTENT_GOLD_OK"
    if g == "UNRESOLVED":
        return "UNRESOLVED_TIMEOUT"
    return "TIMEOUT_MISLABEL"


def main():
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    todo = cohorts()
    print(f"auditing {len(todo)} turns "
          f"({todo.groupby('cohort').size().to_dict()})")
    rows = []
    for r in todo.itertuples():
        theory, goal_q, prev_theory, stored = theory_and_goal(r.example_id, r.turn_index)
        res = rederive_gold(theory, goal_q, P9_BUDGET, M4_BUDGET)
        # pre-edit diagnosis: was Q / ¬Q provable before the edit?
        prev_q, _ = run_prover9(prev_theory, goal_q, P9_BUDGET)
        prev_negq, _ = run_prover9(prev_theory, f"-({goal_q})", P9_BUDGET)
        row = {
            "cohort": r.cohort, "example_id": r.example_id,
            "turn_index": r.turn_index, "transition": r.transition,
            "stored_gold": stored,
            **{k: v for k, v in res.items() if not k.startswith("t_")},
            "prev_q_prover9": prev_q, "prev_negq_prover9": prev_negq,
            "verdict": verdict_of(res, stored),
        }
        rows.append(row)
        print(f"  {r.cohort:20s} {r.example_id:12s} t{r.turn_index} "
              f"{r.transition}  stored={stored:9s} rederived={res['rederived_gold']:12s} "
              f"-> {row['verdict']}")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "tables", "comparator_audit.csv"), index=False)

    print("\nverdicts by cohort:")
    print(out.groupby(["cohort", "verdict"]).size())

    comp = out[out.cohort == "comparator_add_flip"]
    bad = comp.verdict.isin(["INCONSISTENT", "TIMEOUT_MISLABEL"]).sum()
    frac = bad / len(comp) if len(comp) else 0.0
    print(f"\n§1.1 DECISION RULE: {bad}/{len(comp)} comparator turns "
          f"INCONSISTENT or TIMEOUT_MISLABEL ({frac:.0%}); threshold 1/3 -> "
          f"{'DEMOTE P1 DiD to suggestive' if frac >= 1/3 else 'P1 DiD stands'}")


if __name__ == "__main__":
    main()
