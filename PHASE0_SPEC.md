# ReviseQA Phase 0 — Log Mining & Interaction Test

**Task brief for Claude Code.** Drop this file in the repo root, then prompt: *"Read PHASE0_SPEC.md and execute it phase by phase. Stop and ask at any stop-condition."*

---

## 0. Context

ReviseQA v1 evaluated 19 LLMs on 400 examples × 7 edit turns, in two settings — **explicit** (edits stated as deltas) and **implicit** (full rewritten context) — under {COT, Standard} prompting and {feedback, no-feedback}. Published pooled results: additions ≈ 73.6% accuracy vs removals ≈ 50.6%; implicit ≫ explicit.

**Question this phase answers:** is the removal deficit *presentation-dependent* (large in explicit, collapsed in implicit → retracted premises still in the context window are the problem — "suppression" story) or *intrinsic* (same size in both settings → contraction itself is hard)? Plus an error autopsy: when models fail on answer-changing edits, do they keep the stale answer or retreat to Uncertain?

Everything here is offline analysis of existing eval logs + dataset files. **No API calls. No new model runs.**

---

## 1. Hard rules

1. Raw logs and dataset files are **read-only**. All outputs go under `analysis/phase0/`.
2. Fully deterministic: seed 464 everywhere (numpy, bootstrap resampling, any sampling).
3. Python only: pandas, numpy, statsmodels, scipy, matplotlib. No R, no seaborn dependency needed.
4. Never silently improvise around a failed assumption — see §8 stop-conditions.
5. Pin versions in `analysis/phase0/requirements.txt`; provide `run.sh` that reproduces everything end-to-end.

---

## 2. Phase 0.0 — Gate checks (before any analysis)

### 2.1 Harness-bug audit (do this absolutely first)
The paper's Appendix B shows the **Standard Explicit** prompt template ending with `added rules: {{removed_rules}}` — the added-rules slot bound to the removed-rules variable. Find the actual prompt-construction code used in the eval runs. Determine whether this binding exists **in code** or only in the paper's LaTeX.

- Report: file, line, verdict (`CODE_BUG` / `LATEX_TYPO_ONLY`), and which runs are affected.
- **Decision rule:** if `CODE_BUG`, all Standard-explicit runs are contaminated → excluded from primary analysis; the primary prompting strategy becomes **COT** (both settings). If clean, primary = Standard (closest to real chat usage), with COT as robustness.

### 2.2 Data inventory & pairing audit
- Enumerate every run: model × setting × prompting × feedback. Report examples and turns found per run, as a table.
- **Pairing check (critical):** for each model × prompting, verify the *same example IDs* appear in the explicit and implicit runs. Report intersection size. The interaction test is only credible on the paired intersection.
- Reconcile dataset size (paper says 933 in §3.3 and 930 in §4 — report the true count) and confirm the 400-example eval subset is recoverable.
- Confirm per-turn gold answers exist for every turn (from logs or by joining dataset files).

### 2.3 Edit-metadata join
Eval logs may store only predictions. Edit composition lives in the dataset JSON (per edit: `removed_facts`, `removed_rules`, `added_facts`, `added_rules`, each a list of {fol, nl}). Join key assumption: `example_id` + `edit_number`. Verify on 20 random examples by hand-checking that joined metadata matches the prompt content in the explicit logs. Report the verified join key.

### 2.4 Parse-quality audit
- Extract the model's answer per turn from the JSON responses (`answer` field; map A/B/C ↔ True/False/Uncertain; handle both formats).
- Report answer-extraction failure rate per run. Failures are coded `parse_ok = False` and `correct = False` (matching the paper's implicit convention), but retained in the table.
- Models with > 15% failures in any primary run (expect qwen3-235b-a22b-thinking, gpt-5-nano) are **flagged**: kept in per-model outputs, excluded from pooled primary estimates, included in sensitivity.

---

## 3. Phase 0.1 — Tidy table

Build one row per (model, prompting, feedback, setting, example_id, turn):

| column | definition |
|---|---|
| `pred_raw`, `pred` | raw string; normalized ∈ {True, False, Uncertain, PARSE_FAIL} |
| `gold_t`, `gold_prev` | gold answer at turn t and t−1 (t=0 uses the initial QA answer) |
| `correct` | pred == gold_t |
| `parse_ok` | bool |
| `n_add`, `n_rem` | counts of added / removed statements (facts + rules) |
| `edit_class` | `add_only` (n_rem=0, n_add>0), `removal` (n_rem>0), `none` (both 0 — report if this occurs) |
| `edit_target` | fact / rule / both |
| `answer_changed` | gold_t ≠ gold_prev |
| `transition` | e.g. `T→F`, `U→T`, `T→T` |
| `turn_index` | 1–7 |

**Important structural fact to verify, not assume:** v1's Invariant prompt forbids removals, so *pure removals may not exist*; removals should appear almost exclusively inside flips (remove+add) and possibly equivalent-rewrite invariants (remove old form + add equivalent form). Report the full cell-count table `edit_class × answer_changed` before any modeling. The natural 2×2 in this data is:

- `add_only × invariant` — redundant additions
- `add_only × changed` — Uncertain→True edits
- `removal × invariant` — equivalent rewrites (if present)
- `removal × changed` — flips

**Unit tests for the coder** (from the paper's Uriel example): Edit #1 = `add_only`, invariant; Edit #2 = `removal` (rule removed + rule added), changed, target=rule. Add 3 more hand-checked cases from the dataset.

Write `analysis/phase0/tidy.parquet` + `.csv`.

---

## 4. Phase 0.2 — Descriptives

Primary slice: no-feedback, primary prompting strategy (per §2.1), non-flagged models, paired example intersection.

1. Accuracy by `edit_class × setting`, pooled and per model.
2. Accuracy for the full `edit_class × answer_changed × setting` table (the 2×2 above, per setting), pooled and per model.
3. Sanity anchor: reproduce the paper's pooled ≈73.6% (add) vs ≈50.6% (remove) with your coding; report your numbers and explain any deviation (their pooling may include feedback runs or both strategies).
4. **Interaction figure:** small-multiples, one panel per model; x-axis {explicit, implicit}; two lines (add_only vs removal), per-turn conditional accuracy. Parallel = intrinsic; converging = suppression. Also one pooled panel.
5. Per-turn-index accuracy curves by edit_class × setting (checks for degradation-over-turns artifacts).

---

## 5. Phase 0.3 — Interaction tests

All on the primary slice; per model and pooled; item = (example_id, turn); cluster = example_id.

**Pre-registered contrasts:**

- **P1 (primary — answer-changing turns only):** removal-vs-add among answer-changing edits, i.e. flips (`removal × changed`) vs U→T (`add_only × changed`).
  DiD = (Add_expl − Rem_expl) − (Add_impl − Rem_impl), restricted to changed turns. This holds "the answer moved" constant. Residual confound to note in the report: transitions differ (T↔F vs U→T).
- **P2 (all turns):** `edit_class × setting` interaction on all turns, with `answer_changed` and `turn_index` as covariates. Note that `answer_changed` is partly collinear with `edit_class`; report VIFs.
- **P3 (invariant turns only, if `removal × invariant` has n ≥ 200 turns pooled):** rewrites vs redundant additions across settings — operation cost with zero answer change. This is a free preview of the B0-vs-C0 contrast from the matched-generation plan.

**Estimators (run all three, agreement expected):**

1. **DiD on accuracy points** with cluster bootstrap over examples, B = 2000, seed 464 → 95% percentile CI.
2. **GEE logistic** (statsmodels): `correct ~ C(edit_class) * C(setting) + turn_index`, exchangeable working correlation, clustered by example; report interaction coefficient as OR with robust CI.
3. **Paired item deltas:** for each (example, turn), Δ = correct_implicit − correct_explicit; compare mean Δ between removal and add_only turns (cluster-bootstrap CI on the difference of means; Wilcoxon as a check).

**Sensitivity battery:** (a) exclude parse-fail turns instead of scoring them wrong; (b) include flagged models; (c) the excluded prompting strategy, if §2.1 said clean; (d) feedback runs; (e) per edit_target (fact vs rule).

---

## 6. Phase 0.4 — Error autopsy

On the primary slice, for every incorrect turn:

- If `answer_changed`: label **stale** (pred == gold_prev), **uncertain_retreat** (pred == Uncertain, gold_t ≠ Uncertain, gold_prev ≠ Uncertain), else **other_changed**.
- If invariant: label **spurious_to_uncertain** (pred == Uncertain) or **spurious_flip** (pred == negation of gold).

Deliver: counts and shares by `setting × edit_class`, pooled and per model; stacked-bar figure. Headline cells to call out in the report: on flip turns, stale vs uncertain_retreat share, explicit vs implicit. (Paper's Table 5 found 45.6% "excessive uncertainty" overall — this tests whether flips fail by retreating to the non-committed midpoint rather than by keeping the old belief.)

---

## 7. Deliverables

```
analysis/phase0/
  report.md          # bug verdict, inventory, pairing, cell counts, descriptives,
                     # P1–P3 estimates + CIs, autopsy, sensitivity, limitations,
                     # decision-gate readout
  tidy.parquet, tidy.csv
  tables/*.csv       # every table in the report, machine-readable
  figures/*.png+pdf  # interaction small-multiples, per-turn curves, autopsy bars
  src/               # parse_logs.py, code_edits.py, analysis.py, tests/
  run.sh, requirements.txt, README.md
```

**Decision gate (pre-registered; thresholds editable in one place at top of `analysis.py`):**

- **Suppression story GO:** pooled P1 DiD ≥ 5 accuracy points, bootstrap CI excluding 0, and DiD > 0 in ≥ 60% of non-flagged models.
- **Intrinsic-contraction story GO:** implicit-setting removal deficit (Add_impl − Rem_impl on changed turns) ≥ 5 points with CI excluding 0.
- **Uncertain-retreat headline:** uncertain_retreat ≥ 50% of errors on flip turns, pooled.
- Any of the three can pass independently; report all. If none pass, recommendation = proceed to matched generation without the suppression claim.

---

## 8. Stop-and-ask conditions

Stop and report (do not improvise) if: log schema doesn't match assumptions after inspecting 3 runs; per-turn gold answers are unrecoverable; explicit/implicit example-ID overlap < 90% for more than 3 models; the §2.3 join fails hand-verification; `removal × changed` or `add_only × changed` has < 100 pooled turns; or the §2.1 audit is inconclusive.

## 9. Acceptance checklist

- [ ] §2.1 verdict with file+line evidence
- [ ] Pairing table and cell-count table printed in report before any test
- [ ] Coder unit tests pass (5 hand-checked cases)
- [ ] Paper's pooled add/remove split reproduced within noise, or deviation explained
- [ ] P1–P3 with all three estimators + sensitivity battery
- [ ] Autopsy tables + figures
- [ ] `run.sh` reproduces everything from raw logs on a clean checkout
