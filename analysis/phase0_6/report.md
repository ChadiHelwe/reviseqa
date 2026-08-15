# Phase 0.6 — Clean-Slice Re-cut & Valid-Subset Preview: Report

Executed per `PHASE0_6_RECUT.md`; seed 464; inputs read-only; reproduce with
`analysis/phase0_6/run.sh`.

**Setup.** `clean_examples` = manifest-clean ∩ eval-400 = **301**
(expected 301 ✓). `demo_correct` = **193/400**
(expected ~193 ✓); clean ∩ demo_correct = **148**. Primary slice as in
Phase 0 (Standard × no-feedback × 17 non-flagged models × paired ×
actual-delta coding). No stop-conditions triggered.

## Job A — Re-cut citable findings on clean_examples

Hold criteria (`tables/a_hold_criteria.csv`):

| finding | criterion | value | pass |
|---|---|---|---|
| A1 flip autopsy | retreat pooled >= 50% and explicit > implicit | pooled 75.59%, expl 85.5% vs impl 50.1% | True |
| A2 P3 invariant | DiD CI contains 0 | DiD -1.81 [-5.22, 1.65] | True |
| A4 P1b | DiD > 0 | DiD incomputable: add_only×changed comparator empty on clean slice (n_add=0); pure-removal gain nan pts | False |
| A5 transition split | explicit retreat shares within 10 pts | T→F 85.57% vs F→T 85.43% | True |

**Three of four hold criteria pass (A1, A2, A5). A4 FAILS — flagged
prominently, nothing adjusted:** the P1b DiD is *incomputable* on the clean
slice because, once examples are manifest-clean **and** turns are
delta-verified, the changed band is compositionally homogeneous — there are
**zero** `add_only × changed` turns and **zero** pure-removal changed turns
left. Every fully-verified answer-changing edit is a remove+add compound
flip (exactly what FOL monotonicity requires). The apparent compositional
variety in the changed band (pure adds that flip, pure removals that flip)
was *entirely* a metadata/labeling artifact. Consequence: **no
within-changed-band compositional contrast (P1- or P1b-style) is estimable
from v1 data at all** — that comparison genuinely requires Study 2's
generated conditions. For reference, on the original (all-examples) slice
the audit-clean-restricted P1b DiD is +11.6 [−6.6, +34.1] (n.s., 5 unique
turns).

1. **Flip autopsy** (`a1_flip_autopsy.csv`): pooled retreat
   75.59% (original
   74.17%); explicit
   85.5% vs implicit
   50.1% (original
   84.36/50.75).
   Excluding turn 1: pooled 76.53%.
   The finding is unchanged by cleaning.
2. **P3** (`a2_p3_invariant.csv`): clean-slice DiD
   -1.81 [-5.22, 1.65]
   (original -1.32 [-4.34,
   1.68]) — null preserved.
3. **Invariant-band descriptives** (`a3_invariant_descriptives.csv`):
   within ±2 pts of original in every cell.
4. **P1b** (`a4_p1b_audit_clean.csv`): incomputable on the clean slice —
   n_add = n_pure_rem = 0 changed turns survive full verification (see the
   hold-criteria discussion above). Original-slice audit-clean-restricted
   DiD: 11.57 [-6.59,
   34.05] — positive but no longer significant once
   restricted to audited turns. Phase 0's P1b should be cited only with
   this caveat attached.
5. **Transition split** (`a5_transition_split.csv`): explicit retreat shares
   T→F 85.57%
   vs F→T 85.43%
   — symmetric within 10 pts.

Figures: `figures/recut_autopsy_transition.{png,pdf}`.

## Job B — Valid-subset preview (BIASED_SUBSET_PREVIEW — internal only, not for paper)

Subset: 148 examples (logic-clean ∧ demonstration-correct); on these,
existing v1 responses are fully valid end-to-end. Tables:
`b_lcata_preview.csv` (LCATA@2/4/7 per model × setting: published-equivalent
recomputed from full-400 logs with the harness's own scoring, vs the
valid-subset with corrected scoring), `b_per_turn_accuracy.csv`.

Composition bias (`b_composition_bias.csv`, `b_flipcount_distribution.csv`):
demo-correct requires an even number of net flips, so odd-flip trajectories
are excluded by construction —

| in_valid | n | mean_flips | mean_invariant_share |
|---|---|---|---|
| False | 252 | 3.65 | 0.48 |
| True | 148 | 3.46 | 0.51 |

Pooled LCATA deltas (preview − published-equivalent), mean across models:
@2 +4.6, @4 +5.3,
@7 +5.9 pts (explicit and implicit pooled;
per-model table in CSV). Interpretation: order-of-magnitude only — the
subset is easier (more invariant turns) *and* scoring recoveries push in the
same direction, so treat as an upper bound on how much corrected re-runs
would raise published trajectory numbers.

## Job C — Leftover comparator turns

| example_id | turn_index | stored_transition | pre_stored_label | pre_rederived_label | pre_q_prover9 | pre_negq_prover9 | pre_q_countermodel | pre_negq_countermodel | pre_consistency_mace4 | pre_mislabeled | true_transition | flip_illusory |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ex_802 | 6 | T→F | True | Uncertain | EXHAUSTED | EXHAUSTED | MODEL | MODEL | MODEL | True | U→F | True |
| ex_813 | 3 | F→T | False | True | PROVED | EXHAUSTED | EXHAUSTED | MODEL | MODEL | True | T→T | True |

Both surviving `CONSISTENT_GOLD_OK` comparator turns have **mislabeled
pre-edit states** (predicted): the stored pre-edit answers do not re-derive
at 600 s with certificates, so the "flips" are illusory (`ex_802` t6 is
really U→F, not T→F; `ex_813` t3 is really T→T, not F→T). **The Phase-0 P1
comparator is now 18/18 invalid.** Appended to the Phase-0 erratum.

## Job D — Demo-bug impact quantification

1. **Anchoring fingerprint** (`d1_anchoring_fingerprint.csv`): on
   turns 1–2 errors, P(pred == displayed demo answer) on demo-wrong
   examples minus the matched demo-correct rate =
   **+0.7 pts pooled**
   (weak threshold |x| ≤ 5).
2. **Stratified depression** (`d2_stratified_depression_by_turn.csv`):
   accuracy(demo-correct) − accuracy(demo-wrong) within strata matched on
   (flip count, first-flip position) × setting, by turn. Odd-flip strata are
   demo-wrong by parity and unmatchable; coverage per turn is in the table.
3. **Decay:** mean matched depression at turns ≥ 3 =
   **-0.34 pts**
   (threshold ≤ 3).

**Verdict: 301-clean absolute tables with quantified caveat; NO re-runs needed**

## Non-goal (per spec)

No log-derived absolute table in this repo may be presented as bug-free
without the Job D verdict attached. Log surgery removes quarantined
examples, corrupted-delta turns, and scoring errors; it cannot correct model
behavior conditioned on corrupted prompts — Job D bounds that residual. No
fresh-400 re-run is planned (zero budget); Study-2 items were never shown to
any model, so no recomputation substitutes for its new inference.
