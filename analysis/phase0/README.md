# ReviseQA Phase 0 — Log Mining & Interaction Test

Offline, fully deterministic (seed 464) analysis of the existing v1 eval logs,
answering: is the removal deficit *presentation-dependent* (suppression) or
*intrinsic*? Plus an error autopsy of answer-changing turns. No API calls.
Spec: `PHASE0_SPEC.md` (repo root).

## TL;DR (see `report.md` for full detail)

- **Harness audit: LATEX_TYPO_ONLY** — the `added rules: {removed_rules}`
  bug exists only in the paper's LaTeX, not in code (any revision). A
  *different* harness bug was found (demonstration answer bound to the final
  edit's answer; wrong for 52% of examples; setting-neutral).
- **Dataset metadata is corrupted on 18% of turns** (removals absent from
  `edits_made` and hence from explicit prompts). Primary analysis recodes
  edits from actual FOL context deltas and excludes mismatched turns
  (user-approved amendment; pre-registered coding kept as sensitivity f).
- **Suppression gate: GO.** P1 DiD = +15.5 pts [9.0, 21.8], DiD > 0 in 17/17
  models. Removal-flips gain ~20 pts explicit→implicit; addition-flips ~5.
- **Intrinsic gate: NO-GO** (removals are *easier* than the addition-flip
  comparator in both settings).
- **Uncertain-retreat headline: GO** — 74% of flip-turn errors are retreats
  to Uncertain (84% explicit / 51% implicit); stale answers are the minority
  failure mode.
- The paper's pooled 73.6 vs 50.6 add/remove split reproduces only as
  `add_only` vs *pure removals* (~1.7% of turns) — a composition effect.

## Reproduce

```bash
python3 -m pip install -r analysis/phase0/requirements.txt
bash analysis/phase0/run.sh            # or: bash analysis/phase0/run.sh /path/to/python
```

Reads (read-only): `detailed_models_results/`, `reviseqa_data/nl/verified-400/`,
`models_results/`. Writes only under `analysis/phase0/`.
Runtime ≈ 10 min (log parsing + B=2000 cluster bootstraps + GEEs).

## Layout

```
report.md            full report: audits, descriptives, P1–P3, autopsy, gates
tidy.parquet/.csv    one row per (model, prompting, feedback, setting, example, turn)
tables/*.csv         every table in the report, machine-readable
figures/*.png|pdf    interaction small-multiples, per-turn curves, autopsy bars
src/parse_logs.py    builds the tidy table from raw logs + dataset
src/code_edits.py    answer normalization + edit coding (recorded & actual-FOL)
src/analysis.py      descriptives, interaction tests, autopsy, figures, report
                     (decision-gate thresholds at the top)
src/tests/           coder unit tests (5 hand-checked cases)
```

## Key definitions

- **Primary slice**: Standard prompting × no-feedback × 17 non-flagged models
  × paired explicit∩implicit examples × delta-verified turns; edit classes
  from the actual FOL context change.
- **Flagged models** (> 15% parse failures in a primary run):
  qwen-2.5-coder-32b-instruct, qwen3-235b-a22b-thinking-2507.
- **P1 DiD** = (Add_expl − Rem_expl) − (Add_impl − Rem_impl) on
  answer-changing turns; cluster bootstrap over examples, B = 2000, seed 464.
