# Phase 0.6 — Clean-Slice Re-cut & Valid-Subset Preview

Task brief for Claude Code. Amends nothing; consumes `analysis/phase0/tidy.parquet`
and `analysis/phase0_5/v1.1/manifest.csv`. Same hard rules as PHASE0_SPEC §1
(read-only inputs, seed 464, offline, `analysis/phase0_6/` outputs, run.sh).

## Setup

- `clean_examples` = manifest status `clean`, restricted to the 400-example
  eval subset (expected n = 301). Exclude `flagged` and `quarantined`.
- `demo_correct` flag per example: original context answer ==
  `edits[-1]["answer"]` (the answer the buggy harness displayed at turn 0).
  Expected ~193/400 overall; report exact counts for clean ∩ demo_correct.
- Primary slice as in Phase 0: Standard × no-feedback × 17 non-flagged
  models × paired examples × actual-delta coding.

## Job A — Re-cut the citable findings on `clean_examples`

Recompute, each with an `original vs clean-slice` comparison column:

1. Flip-turn autopsy (stale / uncertain_retreat / other) by setting, pooled
   and per model; also excluding turn 1.
2. P3 (invariant rewrites vs redundant additions): cells + DiD + bootstrap CI.
3. Invariant-band descriptives (edit_class × setting accuracy).
4. P1b (pure removals, changed turns): cells + DiD + CI — restrict further to
   the 41 audit-clean pure-removal turns from
   `analysis/phase0_5/tables/comparator_audit.csv`.
5. Transition split (T→F vs F→T gains and retreat shares).

**Hold criteria** (report pass/fail per finding): retreat pooled ≥ 50% and
explicit > implicit retreat share; P3 CI contains 0; P1b DiD > 0; transition
retreat shares within 10 pts of each other. Any failure → flag prominently,
do not adjust anything else.

## Job B — Valid-subset preview (clean ∩ demo_correct)

On examples that are both logic-clean and demonstration-correct, existing v1
responses are fully valid. Compute from existing logs, both settings,
primary slice:

- Per-model LCATA@{2,4,7} and per-turn conditional accuracy.
- Side-by-side with the published Table 3/4 values for the same models.
- Report subset size and its composition bias: distribution of flip counts
  and invariant shares vs the full 400 (expect easier trajectories —
  answer returns to start).

Label every output `BIASED_SUBSET_PREVIEW — internal only, not for paper`.
Purpose: order-of-magnitude estimate of how much corrected re-runs will move
published numbers.

## Job C — Leftover comparator turns

For the 2 `comparator_add_flip` turns that audited `CONSISTENT_GOLD_OK`:
re-derive the *pre-edit* label at 600 s/direction with certificates. Report
whether the pre-edit state was mislabeled (predicted), making the "flip"
illusory. Append the outcome to the Phase-0 erratum block.

## Job D — Demo-bug impact quantification (decides table strategy under zero budget)

The demo bug is in the *prompts* (52% of conversations opened with a wrong
demonstrated answer); re-scoring cannot undo it. Job D measures how much it
actually bit, on `clean_examples`, primary slice:

1. **Anchoring fingerprint:** on demo-wrong examples, at turns 1–2, compute
   P(pred == displayed_demo_answer | error) vs the same quantity on
   demo-correct examples matched on turn-1 edit_class and gold transition.
   Report the excess anchoring rate per model and pooled.
2. **Stratified depression:** accuracy demo-wrong vs demo-correct within
   strata matched on trajectory composition (flip count, invariant share,
   first-flip position). Note that odd-net-flip trajectories are all
   demo-wrong (parity); exclude unmatchable strata and report coverage.
3. **Decay:** does any depression shrink with turn index (anchoring washes
   out) or persist through turn 7?

**Decision rule (state verdict in report.md):**
- Depression ≤ 3 pts at turns ≥ 3 AND weak fingerprint → the paper reports
  301-clean absolute tables with a quantified caveat; **no re-runs needed
  for tables**.
- Otherwise → absolute tables restricted to clean ∩ demo_correct with the
  Job B bias caveat, or moved to appendix as bounded estimates.

## Non-goal (state in report.md)

Do **not** present any log-derived absolute table as bug-free without the
Job D verdict attached. Log surgery legitimately removes quarantined
examples, corrupted-delta turns, and scoring errors; it cannot correct
model behavior conditioned on corrupted prompts — Job D bounds that
residual instead. No fresh-400 re-run is planned (zero-budget constraint);
Study-2 items were never shown to any model, so no recomputation can
substitute for its (new, cheap) inference.

## Deliverables

```
analysis/phase0_6/
  report.md      Jobs A–D results, hold-criteria verdicts, Job D verdict
  tables/*.csv   every table, incl. original-vs-clean comparison columns
  figures/       re-cut autopsy bars, transition split
  run.sh, src/
```

Stop-and-ask if: manifest↔eval-subset join is ambiguous; clean_examples
deviates from 301 by > 5; or demo_correct cannot be computed from stored
initial answers.
