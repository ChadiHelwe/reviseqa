# ReviseQA v1.1 — Changelog

Produced by `analysis/phase0_5/src/repair_v1_1.py` (SPEC_ADDENDUM_A §1.4)
from the 930 verification-preserved v1 examples. Prover9/Mace4 (LADR
2009-11A, Homebrew). Budgets: prover9 60s/direction, mace4
30s, domains ≤ 20.

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

- clean examples: 728
- shipped with label mismatches (flagged): 32
- quarantined (inconsistent or malformed states): 170
