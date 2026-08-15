# ReviseQA Phase 0 — Log Mining & Interaction Test: Report

> **ERRATUM (Phase 0.5, §1.1 comparator audit — SPEC_ADDENDUM_A).** The P1
> comparator (`add_only × changed`, 18 delta-verified turns) failed the
> prover consistency audit: 11/18 post-edit theories are **inconsistent**
> (Prover9 derives a contradiction; the gold label is ill-defined) and 5/18
> are **mislabeled** (re-derived gold at 10× budget differs from the stored
> answer). Only 2/18 are sound. Per the pre-registered decision rule (≥ 1/3),
> **the P1 DiD of +15.5 pts is demoted to *suggestive* and must not be cited
> as a finding.** See `analysis/phase0_5/report.md`.
> *(Evidence-status per SPEC_ADDENDUM_B: the citable Phase-0 results are the
> error autopsy, P3's answer-relevance null, and the descriptives. **P1b is
> also not citable as evidence** — audit-clean restriction +11.6
> [−6.6, +34.1], n.s., 5 unique turns; clean slice incomputable. Canonical
> list: `analysis/EVIDENCE_STATUS.md`.)*
>
> **Phase 0.6 Job C update: the comparator is 18/18 invalid.** The two turns
> that audited CONSISTENT_GOLD_OK have prover-certified *mislabeled pre-edit
> states* (`ex_802` t6 is U→F, not T→F; `ex_813` t3 is T→T, not F→T), so
> those "flips" are illusory. Moreover, on fully verified turns the changed
> band is compositionally homogeneous (all flips are remove+add), so no
> P1/P1b-style within-changed contrast is estimable from v1 at all, and P1b
> weakens to non-significant under audit restriction (+11.6 [−6.6, +34.1]).
> See `analysis/phase0_6/report.md`.

Deterministic analysis (seed 464, bootstrap B=2000) of existing eval
logs; no API calls. Reproduce with `analysis/phase0/run.sh`. All decision-gate
thresholds sit at the top of `analysis/phase0/src/analysis.py`.

> **Pre-registration amendment (user-approved during execution).** The §2.3
> audit was extended to the FOL level and found that the dataset's recorded
> `edits_made` metadata does not match the actual context change on
> 510/2800 example-turns (§1.5). On those turns the explicit
> track showed models a wrong/incomplete delta while the implicit track showed
> the true state — the two settings differ in *content*, not just
> presentation. The primary analysis therefore (i) recodes every turn from the
> **actual FOL context delta** and (ii) **excludes delta-mismatch turns** from
> the interaction tests. The pre-registered spec coding (recorded metadata,
> mismatches included) is reported unchanged as sensitivity (f); it is the
> single specification that materially disagrees, and §4.6 shows why.

## 1. Gate checks (Phase 0.0)

### 1.1 Harness-bug audit (§2.1) — verdict: **LATEX_TYPO_ONLY**

Prompt construction lives in `src/evaluation.py` (`LogicDataset.read_dir`,
current lines 247–265). Each explicit-delta slot is bound to its matching
variable — `"Added rules:"` joins `delta["added_rules"]`
(`src/evaluation.py:257-260`), `"Removed rules:"` joins
`delta["removed_rules"]` (`:253-256`), likewise facts. Checked in **every git
revision** of the file (78e632f → 2382150, May–Sep 2025, spanning the run
dates 2025-09-25..28): correctly bound throughout. Verified empirically too:
logged prompt contexts match the correctly-bound construction
character-for-character on 20 seeded examples × 3 models × 7 turns (420/420,
§1.3). The `added rules: {removed_rules}` binding in the paper's Appendix B
exists only in the LaTeX.

**Decision rule applied:** clean → primary prompting = **Standard**
(`*_no_reasoning` tracks), COT as robustness. Primary slice = Standard ×
no-feedback.

**Separate harness bug found (outside §2.1 scope, reported for completeness):**
`src/evaluation.py:214` sets the turn-0 demonstration answer to
`edits[-1]["answer"]` — the answer after the *final* edit — instead of the
original context's answer. They differ in **207/400** eval examples, so ~52%
of conversations open with a demonstration asserting a wrong answer for the
context shown (in COT tracks it also contradicts the demonstrated reasoning
chain). This is identical across all tracks and settings, so it cannot mimic
an edit-class × setting interaction, but absolute accuracies are depressed by
it and turn-1 autopsy labels carry a caveat (§5).

### 1.2 Data inventory & pairing (§2.2)

19 models × 8 tracks (setting × prompting × feedback) × 400 examples × 7
turns; `google/gemini-2-5-flash` is missing 1 example in 2 tracks (333).
Full inventory: `tables/inventory.csv`. Pairing on the primary runs
(explicit ∩ implicit example IDs):

| model | n_explicit | n_implicit | n_paired |
|---|---|---|---|
| anthropic/claude-sonnet-4 | 400 | 400 | 400 |
| google/gemini-2-5-flash | 400 | 399 | 399 |
| google/gemini-2.5-pro | 400 | 400 | 400 |
| google/gemma-3-12b-it | 400 | 400 | 400 |
| google/gemma-3-27b-it | 400 | 400 | 400 |
| google/gemma-3-4b-it | 400 | 400 | 400 |
| moonshotai/kimi-k2-0905 | 400 | 400 | 400 |
| openai/gpt-4.1-mini | 400 | 400 | 400 |
| openai/gpt-5-nano | 400 | 400 | 400 |
| openai/gpt-oss-120b | 400 | 400 | 400 |
| openai/gpt-oss-20b | 400 | 400 | 400 |
| qwen/qwen-2.5-coder-32b-instruct | 400 | 400 | 400 |
| qwen/qwen3-235b-a22b-2507 | 400 | 400 | 400 |
| qwen/qwen3-235b-a22b-thinking-2507 | 400 | 400 | 400 |
| qwen/qwen3-30b-a3b | 400 | 400 | 400 |
| qwen/qwen3-30b-a3b-thinking-2507 | 400 | 400 | 400 |
| qwen/qwen3-coder | 400 | 400 | 400 |
| qwen/qwen3-coder-30b-a3b-instruct | 400 | 400 | 400 |
| x-ai/grok-code-fast-1 | 400 | 400 | 400 |

Worst overlap 333/334 (99.7%) — far above the 90% stop threshold. All
analyses use each model's paired intersection.

**Dataset size reconciliation:** `reviseqa_data/nl/dataset_statistics.json`
records 1731 generated examples and **930 preserved after verification** —
§4's 930 is correct; §3.3's "933" is stale. The 400-example eval subset is
exactly `reviseqa_data/nl/verified-400/` (334 `ex_*` + 66 `prev_ex_*`), the
directory `evaluate_models.sh` feeds the harness. Per-turn gold answers exist
in both the logs (`correct_answer`) and the dataset (`edits[i].answer`) and
agree on all 425,586 turn rows.

### 1.3 Edit-metadata join (§2.3)

Join key: **`example_id` (from the log filename) + edit position (log `step`
i ↔ dataset `edits[i-1]`; equals `edit_number`)**. Hand-verified on 20
seed-464 random examples × 3 models (gpt-4.1-mini, claude-sonnet-4,
qwen3-30b-a3b): explicit log context == delta rebuilt from `edits_made`,
implicit log context == `edited_natural_language_context`, log gold ==
dataset answer, for all 420 triples. **420/420 passed** — the logs faithfully
reflect the recorded metadata. (§1.5 shows the recorded metadata itself is
what fails against the FOL state.)

### 1.4 Parse-quality audit (§2.4)

Answers re-extracted per spec (JSON `answer` field; A/B/C ↔
True/False/Uncertain; both formats; empty/`ERROR` → `PARSE_FAIL`, scored
incorrect, retained). Parse-fail % in primary runs:

| model | explicit_parse_fail_pct | implicit_parse_fail_pct |
|---|---|---|
| anthropic/claude-sonnet-4 | 0.0 | 0.0 |
| google/gemini-2-5-flash | 0.79 | 0.47 |
| google/gemini-2.5-pro | 0.0 | 0.0 |
| google/gemma-3-12b-it | 0.11 | 0.39 |
| google/gemma-3-27b-it | 0.0 | 0.0 |
| google/gemma-3-4b-it | 0.0 | 0.0 |
| moonshotai/kimi-k2-0905 | 0.25 | 0.18 |
| openai/gpt-4.1-mini | 0.0 | 0.0 |
| openai/gpt-5-nano | 0.04 | 0.07 |
| openai/gpt-oss-120b | 0.04 | 0.0 |
| openai/gpt-oss-20b | 0.21 | 0.29 |
| qwen/qwen-2.5-coder-32b-instruct | 39.61 | 13.43 |
| qwen/qwen3-235b-a22b-2507 | 0.0 | 0.0 |
| qwen/qwen3-235b-a22b-thinking-2507 | 72.43 | 38.61 |
| qwen/qwen3-30b-a3b | 0.11 | 0.04 |
| qwen/qwen3-30b-a3b-thinking-2507 | 0.0 | 0.0 |
| qwen/qwen3-coder | 0.0 | 0.0 |
| qwen/qwen3-coder-30b-a3b-instruct | 0.0 | 0.0 |
| x-ai/grok-code-fast-1 | 0.29 | 0.32 |

**Flagged (> 15% in a primary run):**
`qwen/qwen-2.5-coder-32b-instruct`, `qwen/qwen3-235b-a22b-thinking-2507`. The spec expected `gpt-5-nano` too;
its primary-run failure rate is < 0.1%, so it is **not** flagged. Flagged
models: kept per-model, excluded from pooled primary, included in
sensitivity (b).

Scoring note: the A/B/C mapping recovers answers the harness scored wrong
(bare option letters). All scoring disagreements with the harness's own
`correct` flag are one-directional recoveries (up to 10% of turns in the
worst qwen3-30b-a3b run), so published accuracies modestly underestimate
letter-answering models.

### 1.5 FOL delta-integrity audit (new; basis of the amendment)

Comparing each turn's recorded `edits_made` against the actual FOL context
change (`edited_context_fol[t] − edited_context_fol[t−1]`):
**510/2800 example-turns (18.2%) mismatch**; 509 turns
contain removals absent from the metadata (and hence absent from the explicit
prompt), 494 contain unrecorded additions; a smaller number of recorded edits
never happened ("phantom"). Reclassification recorded → actual:

| edit_class_recorded | edit_class | n_example_turns |
|---|---|---|
| add_only | add_only | 713 |
| add_only | none | 3 |
| add_only | removal | 158 |
| none | none | 8 |
| none | removal | 4 |
| removal | add_only | 5 |
| removal | removal | 1909 |

Notably, **158 recorded-`add_only` turns actually contain removals** — e.g.
`ex_1011` turn 6 silently drops a biconditional from the implicit context
while the explicit delta mentions only the addition. Also ~21% of recorded
"facts" are syntactically rules; actual-delta coding classifies fact/rule
from FOL syntax. Audit table: `tables/fol_delta_audit.csv`.

## 2. Tidy table (Phase 0.1)

`tidy.parquet` / `tidy.csv`: 425,586 rows, one per (model, prompting,
feedback, setting, example_id, turn), with both codings (`edit_class` =
actual-delta; `edit_class_recorded` = spec) and the `delta_mismatch` flag.
Coder unit tests (5 hand-checked cases incl. the paper's Uriel example =
`ex_2512`): **all pass** (`src/tests/test_coder.py`).

Cell counts, primary slice (17 non-flagged models, Standard,
no-feedback, paired, delta-verified turns), per setting:

| edit_class | answer_changed | setting | n_turns |
|---|---|---|---|
| add_only | False | explicit | 11641 |
| add_only | False | implicit | 11641 |
| add_only | True | explicit | 306 |
| add_only | True | implicit | 306 |
| removal | False | explicit | 6341 |
| removal | False | implicit | 6341 |
| removal | True | explicit | 20500 |
| removal | True | implicit | 20500 |

Structural facts **verified, not assumed** — three of the spec's assumptions
fail in the data:

1. **Pure removals exist** (48 unique example-turns under recorded coding;
   more under actual coding) — the Invariant prompt's no-removal instruction
   was not binding. They are ~2% of turns but drive the paper's headline
   removal number (§3).
2. **There are no U→T edits.** The spec's natural 2×2 assumed
   `add_only × changed` = Uncertain→True. In reality *every* answer-changing
   turn is a T↔F flip (~50/50 both directions, one U→F), in both classes.
   `add_only × changed` = flips achieved by pure addition (the gold labels
   are non-monotonic because "False" = ¬conclusion provable). This *removes*
   the pre-registered transition confound — both P1 cells share the same
   transition mix — at the cost of a small comparator (18 unique
   delta-verified example-turns).
3. A small **`none` class** exists (empty actual delta; excluded).

## 3. Descriptives (Phase 0.2)

Accuracy by `edit_class × setting` (primary slice, all turns):

| edit_class | setting | acc | n |
|---|---|---|---|
| add_only | explicit | 57.1 | 11947 |
| add_only | implicit | 87.08 | 11947 |
| removal | explicit | 62.78 | 26841 |
| removal | implicit | 85.29 | 26841 |

Full `edit_class × answer_changed × setting` table:

| edit_class | answer_changed | setting | acc | n |
|---|---|---|---|---|
| add_only | False | explicit | 57.65 | 11641 |
| add_only | False | implicit | 88.29 | 11641 |
| add_only | True | explicit | 36.27 | 306 |
| add_only | True | implicit | 41.18 | 306 |
| removal | False | explicit | 59.19 | 6341 |
| removal | False | implicit | 88.5 | 6341 |
| removal | True | explicit | 63.89 | 20500 |
| removal | True | implicit | 84.3 | 20500 |

Per-model versions: `tables/desc_*_per_model.csv`.

### Sanity anchor — the paper's pooled 73.6 / 50.6

| definition | acc | n |
|---|---|---|
| recorded add_only, all runs, harness scoring | 73.59 | 132840 |
| recorded pure removal (n_rem>0, n_add=0), all runs, harness scoring | 50.09 | 7295 |
| recorded any-removal (n_rem>0, incl. flips), all runs, harness scoring | 71.57 | 290922 |
| recorded add_only, all runs, our scoring (A/B/C mapped) | 74.87 | 132840 |
| recorded pure removal, all runs, our scoring (A/B/C mapped) | 50.93 | 7295 |

The paper's pooled numbers reproduce **exactly** for additions (73.6 =
recorded `add_only`, all runs, harness scoring) and within 0.5 pts for
removals (50.1 vs 50.6) — **but only when "removals" means *pure* removals**
(n_rem>0, n_add=0; ~1.7% of turn-rows; the residual is consistent with the
published pooling including `gpt-5-mini`, whose detailed logs are absent from
this repo). Any-removal turns score 71.6% — indistinguishable from additions.
**The published add-vs-remove gap is a composition effect**: recorded
`add_only` turns are 97% invariant (easy), pure-removal turns are 67%
answer-changing (hard), and pure removals collapse specifically in the
explicit setting (25.3% explicit vs 74.8% implicit, all runs).

### Figures

- `figures/interaction_small_multiples.{png,pdf}` — per-model panels, all
  turns: both classes gain ~25–30 pts explicit→implicit; near-parallel at
  this altitude because invariant turns dominate.
- `figures/interaction_small_multiples_changed.{png,pdf}` — the same
  restricted to answer-changing turns: in nearly every panel the removal
  line rises steeply from explicit to implicit while the add line stays
  almost flat (the suppression signature, model-by-model).
- `figures/interaction_changed_pooled.{png,pdf}` — pooled changed-turn
  panel with the pure-removal line.
- `figures/turn_curves.{png,pdf}` — per-turn curves: explicit accuracy
  drifts down over turns for both classes; implicit stays flat; no
  class-selective turn artifact.

## 4. Interaction tests (Phase 0.3)

Item = (example, turn); cluster = example; DiD = (Add_expl − Rem_expl) −
(Add_impl − Rem_impl). **Positive DiD = removals hurt more by the explicit
setting than additions = suppression signature.**

### 4.1 P1 — primary (answer-changing turns only)

Cell accuracies: Add_expl 36.3, Rem_expl
63.9, Add_impl 41.2,
Rem_impl 84.3
(n = 306 / 20500
add/removal turns per setting).

| estimand | estimate [95% CI] |
|---|---|
| Add − Rem, explicit | -27.62 pts [95% CI -42.21, -10.64] |
| Add − Rem, implicit | -43.13 pts [95% CI -60.48, -24.61] |
| **DiD (P1)** | **+15.51 pts [95% CI +9.02, +21.84]** |

Two separate findings live here:

1. **No intrinsic removal deficit — the sign reverses.** Among
   answer-changing edits, removal-flips are *easier* than addition-flips in
   both settings (the "Add − Rem deficit" is large and negative). The hard
   class is flips-by-pure-addition, whose non-monotonic label semantics
   models handle near chance (36/41%).
2. **A clean suppression signature.** Removal-flips gain
   20.4 pts
   from explicit→implicit while addition-flips gain only
   4.9 pts:
   DiD +15.51 pts [95% CI +9.02, +21.84], and **DiD > 0 in
   100.0% of the 17 non-flagged models**
   (range +5.6 to
   +40.5; `tables/p1_per_model.csv`):

| model | deficit_explicit | deficit_implicit | did | did_ci_lo | did_ci_hi |
|---|---|---|---|---|---|
| anthropic/claude-sonnet-4 | -43.37 | -59.2 | 15.84 | 3.86 | 33.19 |
| google/gemini-2-5-flash | -33.88 | -48.99 | 15.12 | -24.25 | 54.84 |
| google/gemini-2.5-pro | -28.86 | -40.96 | 12.11 | 5.08 | 25.25 |
| google/gemma-3-12b-it | -12.02 | -27.78 | 15.75 | -7.51 | 37.02 |
| google/gemma-3-27b-it | -24.71 | -41.96 | 17.25 | -1.82 | 38.25 |
| google/gemma-3-4b-it | -13.02 | -25.87 | 12.85 | -0.65 | 20.82 |
| moonshotai/kimi-k2-0905 | -17.74 | -42.62 | 24.88 | -0.99 | 49.97 |
| openai/gpt-4.1-mini | -36.65 | -44.36 | 7.71 | -14.67 | 29.97 |
| openai/gpt-5-nano | -5.06 | -45.52 | 40.46 | 13.96 | 65.7 |
| openai/gpt-oss-120b | -35.49 | -48.26 | 12.77 | -0.63 | 20.61 |
| openai/gpt-oss-20b | -16.92 | -40.3 | 23.38 | 2.91 | 43.21 |
| qwen/qwen3-235b-a22b-2507 | -29.27 | -39.22 | 9.95 | -10.17 | 29.6 |
| qwen/qwen3-30b-a3b | -20.65 | -40.63 | 19.98 | -2.15 | 42.0 |
| qwen/qwen3-30b-a3b-thinking-2507 | -38.23 | -48.51 | 10.28 | -14.35 | 33.9 |
| qwen/qwen3-coder | -41.79 | -47.43 | 5.64 | -7.7 | 13.58 |
| qwen/qwen3-coder-30b-a3b-instruct | -19.9 | -33.25 | 13.35 | -10.6 | 37.83 |
| x-ai/grok-code-fast-1 | -51.99 | -58.29 | 6.3 | -12.99 | 26.79 |

### 4.2 P1b — supplementary: *pure* removals vs additions (changed turns)

Cells: PureRem_expl 30.7, PureRem_impl
48.4 (n = 153
per setting; only 9 unique delta-verified example-turns — descriptive only).
DiD = +12.75 pts [95% CI +0.73, +26.64] — same direction as P1, wide CI.

### 4.3 P2 — GEE logistic

`correct ~ C(edit_class, Treatment('add_only')) * C(setting, Treatment('explicit')) + answer_changed + turn_index`, exchangeable working
correlation, clustered by example.

| slice | interaction OR (removal × implicit) [95% robust CI] | p | n |
|---|---|---|---|
| all turns | 0.676 [0.580, 0.787] | 4.6e-07 | 77,576 |
| changed turns only | 2.526 [1.924, 3.317] | 2.5e-11 | 41,612 |

**The two GEE rows disagree in direction, and the all-turns row should not be
read as the P1 check.** On changed turns only — the P1 estimand — the GEE
agrees with the bootstrap DiD: OR 2.53 > 1
(removals benefit more from implicit). The all-turns OR < 1 is a
composition/scale artifact: `add_only` mass sits on invariant turns
(57.7→88.3, log-odds gain 1.71) while `removal` mass sits on changed turns
(63.9→84.3, log-odds gain 1.11), and a single additive `answer_changed` term
cannot absorb the class-specific ceiling geometry. Per-model all-turns ORs:
`tables/p2_gee_per_model.csv`.

VIFs for the P2 design matrix — `answer_changed` is only mildly collinear
with `edit_class` (all < 1.9):

| term | vif |
|---|---|
| C(edit_class, Treatment('add_only'))[T.removal] | 1.89 |
| C(setting, Treatment('explicit'))[T.implicit] | 1.0 |
| answer_changed_int | 1.88 |
| turn_index | 1.01 |

### 4.4 P3 — invariant turns (rewrites vs redundant additions)

`removal × invariant` pooled n = 12,682 ≥ 200
→ run. Cells: Add_expl 57.6, Rem_expl
59.2, Add_impl 88.3,
Rem_impl 88.5.

| estimand | estimate [95% CI] |
|---|---|
| Operation cost, explicit (Add−Rem) | -1.54 pts [95% CI -4.83, +1.71] |
| Operation cost, implicit (Add−Rem) | -0.21 pts [95% CI -2.49, +2.13] |
| **DiD (P3)** | **-1.32 pts [95% CI -4.34, +1.68]** |

**Null.** With zero answer movement, rewrite-removals cost nothing relative
to redundant additions in either setting. The suppression effect (P1) is
specific to turns where the removal *matters* for the answer — retracted-but-
visible premises hurt only when the conclusion depends on the retraction.
This is the free preview of the matched-generation B0-vs-C0 contrast:
expect ≈ 0 there too unless the edit is answer-relevant.

### 4.5 Estimator 3 — paired item deltas (changed turns)

Δ = correct_implicit − correct_explicit per (model, example, turn). Mean Δ:
add_only +0.049 (n = 306),
removal +0.204 (n = 20,500);
difference (removal − add) = **+0.155
[95% cluster-bootstrap CI +0.090, +0.218]**,
Mann-Whitney p = 1.3e-07. Matches P1's DiD
(+15.5 pts) — estimators 1 and 3 agree exactly;
estimator 2 agrees on the matching (changed-only) estimand (§4.3).

### 4.6 Sensitivity battery

| slice | deficit_explicit | deficit_implicit | did | did_ci_lo | did_ci_hi |
|---|---|---|---|---|---|
| primary | -27.62 | -43.13 | 15.51 | 9.02 | 21.84 |
| a_exclude_parse_fail | -27.71 | -43.21 | 15.51 | 9.05 | 21.82 |
| b_include_flagged | -26.77 | -42.62 | 15.86 | 9.64 | 22.01 |
| c_cot_prompting | -29.72 | -45.41 | 15.7 | 8.61 | 21.79 |
| d_feedback_runs | -33.67 | -44.97 | 11.3 | 3.23 | 18.49 |
| f_prereg_spec_coding | -28.44 | -30.61 | 2.16 | -9.01 | 12.6 |
| g_actual_coding_incl_mismatch | -22.3 | -36.35 | 14.05 | 5.94 | 21.75 |
| e_target_fact | -20.07 | -36.38 | 16.31 | 0.97 | 31.76 |
| e_target_rule | -30.56 | -48.12 | 17.56 | 9.51 | 24.63 |
| e_target_both |  |  |  |  |  |

(`e_target_both` skipped: add_only × changed cell < 50 turns.)

The DiD is stable at +11 to +18 with CI excluding 0 under: parse-fail
exclusion (a), flagged-model inclusion (b), COT prompting (c), feedback (d),
actual coding *with* mismatched turns kept (g), and within fact-targeted and
rule-targeted edits (e). **The only specification that kills it is the
pre-registered one (f): recorded metadata coding with mismatched turns
included (DiD +2.2, CI spans 0).** The difference between (f) and (g)/(primary)
is precisely the 510 metadata-corrupted turns — mislabeled removals
sitting in the add_only comparator and explicit-track deltas that misdescribe
the state. The pre-registered null is an artifact of dataset metadata errors,
not evidence against suppression.

## 5. Error autopsy (Phase 0.4)

All incorrect turns in the primary slice, labeled per §6 of the spec.
Tables: `tables/autopsy_pooled.csv` (+ per-model); figure:
`figures/autopsy_bars.{png,pdf}`.

**Headline — flip-turn (removal × changed) error composition:**

| setting | stale | uncertain_retreat | other |
|---|---|---|---|
| explicit | 15.3% | 84.4% | 0.4% |
| implicit | 48.6% | 50.7% | 0.7% |

Pooled: **uncertain_retreat = 74.2%**,
stale = 25.4%. Excluding turn 1 (demo-answer caveat):
retreat 75.0%, stale
24.6% — robust.

Models that fail flips overwhelmingly **retreat to Uncertain rather than keep
the stale answer**, and the retreat is concentrated in the explicit setting
(84% of explicit flip errors vs 51% implicit). Invariant-turn errors show the
same signature: ~84% of explicit invariant errors are spurious collapses to
Uncertain (vs ~46-53% implicit). This sharpens the paper's Table 5 (45.6%
"excessive uncertainty"): delta-presentation drives models toward
non-commitment, not belief perseverance — stale answers are the *minority*
failure mode everywhere.

## 6. Decision-gate readout (pre-registered thresholds, amended slice)

| gate | criterion | value | pass |
|---|---|---|---|
| suppression_GO | P1 DiD >= 5.0 pts, CI excl 0, >= 60% models DiD>0 | DiD=+15.51 [+9.02,+21.84], models>0: 100% | True |
| intrinsic_contraction_GO | implicit deficit (Add_impl - Rem_impl, changed) >= 5.0 pts, CI excl 0 | -43.13 [-60.48,-24.61] | False |
| uncertain_retreat_headline | uncertain_retreat >= 50% of flip-turn errors | 74.2% (stale: 25.4%) | True |

- **Suppression story: GO.** P1 DiD +15.51 pts [95% CI +9.02, +21.84], ≥ 5 pts, CI
  excludes 0, and DiD > 0 in 100% of non-flagged models. On the
  pre-registered (uncorrected) coding the gate fails (+2.2, CI spans 0) —
  §4.6 attributes the difference to metadata corruption.
- **Intrinsic-contraction story: NO-GO** — the implicit removal deficit is
  large and *negative* (removals are easier than the addition-flip
  comparator once "the answer moved" is held constant).
- **Uncertain-retreat headline: GO** (74.2%
  ≥ 50%).

**Recommendation:** proceed to matched generation *with* the suppression
claim, framed precisely: retracted-premises-in-context hurt on
answer-relevant removals (P1), cost nothing on answer-irrelevant ones (P3),
and the behavioral failure mode is retreat-to-Uncertain, not stale-belief
persistence (§5). Carry two caveats into the design: the addition-flip
comparator is small (18 delta-verified example-turns) and semantically
unusual (non-monotonic label flips); and the dataset's edit metadata needs
regeneration before any further explicit-setting evaluation.

## 7. Limitations

1. **Post-hoc amendment**: the primary slice recodes edits from actual FOL
   deltas and excludes 18.2% of turns; the pre-registered coding is
   sensitivity (f). The amendment was forced by verifiable metadata errors
   and approved mid-run, but it is not pre-registered inference.
2. **Small comparator**: `add_only × changed` has 18 unique delta-verified
   example-turns (306 items/setting pooled). Cluster-bootstrap CIs account
   for the clustering, but per-model add-cell estimates rest on ≤ 18 items.
3. **Comparator semantics**: flips-by-pure-addition rely on "False =
   ¬conclusion provable" non-monotonicity; they may be intrinsically harder
   than removal flips for reasons unrelated to presentation, which the DiD
   only partially nets out.
4. **Demonstration-answer bug** (§1.1): setting-neutral, but absolute
   accuracies are depressed and turn-1 gold_prev uses the true initial
   answer, not the (sometimes wrong) demonstrated one; autopsy is robust to
   dropping turn 1.
5. **Pooled GEE clusters only by example**, treating model repeats within an
   example as exchangeable; per-model GEEs and the bootstrap agree with the
   pooled conclusion on the changed-only estimand.
6. `implicit_shuffled` tracks exist for no model in this dump; that
   robustness check is not reproducible here.

## 8. Acceptance checklist

- [x] §2.1 verdict with file+line evidence (LATEX_TYPO_ONLY;
      `src/evaluation.py:249-265`, all git revisions + empirical check)
- [x] Pairing table and cell-count table printed before any test (§1.2, §2)
- [x] Coder unit tests pass (5 hand-checked cases + normalization suite)
- [x] Paper's pooled add/remove split reproduced (73.6 exact; 50.1 vs 50.6
      explained: pure-removal definition + absent gpt-5-mini logs)
- [x] P1–P3 with all three estimators + sensitivity battery (a–g)
- [x] Autopsy tables + figures
- [x] `run.sh` reproduces everything from raw logs on a clean checkout
