# Phase 0.5 — Verification Follow-ups (SPEC_ADDENDUM_A §1): Report

> **Evidence-status update (SPEC_ADDENDUM_B, after Phase 0.6).** Statements
> below that P1b "remains citable" are superseded: Phase 0.6 found the
> changed band compositionally homogeneous on fully verified turns (P1b
> incomputable on the clean slice; +11.6 [−6.6, +34.1] n.s. under audit
> restriction), and Job C showed the 2 surviving comparator flips are
> illusory (comparator 18/18 invalid). Canonical citable list:
> `analysis/EVIDENCE_STATUS.md`.

Executed 2026-08-08. Deterministic; prover work via Prover9/Mace4 (LADR
2009-11A, Homebrew build). Reproduce with `analysis/phase0_5/run.sh`
(requires `analysis/phase0/tidy.parquet` and the prover binaries).
Budgets: gold re-derivation at 600 s/direction (10× the pipeline's nltk
default of 60 s) for §1.1/§1.3; 60 s/direction for the full-dataset §1.4
sweep; Mace4 at 30–60 s, domains ≤ 20.

**Toolchain validation.** The FOL→LADR converter was validated against every
stored `prover9_input` in verified-400: 2607/2800 edits match up to syntax
style; of the 193 that differ, 125 are prover9-provably theory-equivalent,
**54 are provably NOT equivalent** (the stored prover input differs
semantically from the displayed context), and 14 were initially unresolved
(raw `≠`/unicode in stored inputs; resolved after mapping). Audits therefore
re-derive all labels from the *displayed* context (`edited_context_fol`),
not the stored prover input. This is a fourth data-integrity finding beyond
Phase 0's three.

## §1.1 Comparator consistency audit — **P1 comparator is broken; DiD demoted**

Cohorts audited (`tables/comparator_audit.csv`): the 18 delta-verified
`add_only × changed` turns (the P1 comparator), the 1 U→F turn, and the 48
recorded pure-removal turns. Per turn: Mace4 model search, Prover9
contradiction search, gold re-derivation both directions at 10× budget with
Mace4 countermodel certificates, plus the pre-edit state for diagnosis.

| cohort | n | CONSISTENT_GOLD_OK | INCONSISTENT | TIMEOUT_MISLABEL | UNRESOLVED |
|---|---|---|---|---|---|
| comparator_add_flip | 18 | 2 | **11** | **5** | 0 |
| u_to_f | 1 | 1 | 0 | 0 | 0 |
| pure_removal | 48 | 41 | 3 | 3 | 1 |

The monotonicity argument was correct: a T↔F flip by pure addition entails an
inconsistent post-edit theory unless mislabeled, and 11/18 comparator
theories indeed prove a contradiction (gold ill-defined), while 5 more
re-derive to a different label than stored. **16/18 (89%) ≥ 1/3 →
per the pre-registered decision rule, the P1 DiD (+15.5) is demoted to
"suggestive" everywhere.** An erratum block now heads
`analysis/phase0/report.md` (and its generator template). Citable Phase-0
results remain: P1b (pure removals, 41/48 audit-clean), the autopsy
asymmetry, P3's answer-relevance null, the invariant-band descriptives, and
the composition-effect diagnosis of the published 73.6/50.6.

## §1.2 Transition split — retreat is direction-symmetric

`tables/transition_split.csv`, `figures/transition_split.{png,pdf}`.
Removal × changed, primary slice:

| transition | acc explicit | acc implicit | gain | retreat share of explicit errors | retreat share of implicit errors |
|---|---|---|---|---|---|
| T→F | 63.7 | 82.0 | +18.3 | 84.3% | 46.0% |
| F→T | 64.0 | 86.6 | +22.5 | 84.5% | 57.1% |

Prediction confirmed: retreat-to-Uncertain dominates explicit-setting errors
**identically in both directions** (84.3% vs 84.5%), ruling out "False is
just hard to say." The explicit→implicit gain is likewise similar in both
directions. This supports the recommitment-failure vocabulary (§3 of the
addendum).

## §1.3 Timeout-Uncertain audit — clean, and U is structurally absent

Only **2 gold-U states exist in all of verified-400** (both initial states:
`ex_2712`, `prev_ex_352`; zero post-edit U states). Both re-verify as
`U_CONFIRMED_COMPLETED`: Prover9 exhausts both directions *and* Mace4
produces countermodels both ways. **TIMEOUT_AS_U rate = 0/2**
(`tables/timeout_u_audit.csv`).

The load-bearing consequence is the near-absence itself: v1 cannot support
any gold-U condition (C1's gold-U items must be *generated* in Study 2, per
addendum §2.4), and the "Uncertain" option in the eval was effectively
always a wrong answer — context for the retreat-to-Uncertain failure mode.

## §1.4 v1.1 repair — frozen at `analysis/phase0_5/v1.1/`

Applied to the 930 verification-preserved examples
(`verification_summary.csv`, majority=True):

- **(a) Harness fix applied**: `src/evaluation.py` demonstration answer now
  binds to the original context's answer (was `edits[-1]["answer"]`, wrong
  for 207/400 eval examples).
- **(b) `edits_made` regenerated** from actual FOL context diffs (NL via the
  parallel NL/FOL arrays, fact/rule by FOL syntax); v1 field preserved as
  `edits_made_v1`.
- **(c) Mace4 consistency stamp** on all ~7,400 turn states.
- **(d) Prover-budget metadata** per direction per turn (proved / exhausted /
  timeout + countermodel certificates + budgets), `rederived_label`, and
  `label_agrees_with_stored`.

Outcome (`v1.1/manifest.csv`, `v1.1/CHANGELOG.md`):

| status | examples |
|---|---|
| clean | 728 |
| shipped with label-mismatch flags (excl. from Study-2 sampling) | 32 |
| **quarantined (≥ 1 provably inconsistent turn state)** | **170 (18.3%)** |

263 turn states across 170 examples are inconsistent — the quarantine rate
matches Phase 0's 18.2% metadata-corruption rate, confirming a single
underlying pipeline defect (edits applied to the context without being
recorded, and vice versa). Of the 400-example eval subset: 301 clean, 18
flagged, **81 quarantined** — v1 results on ~20% of eval items were scored
against ill-defined or unverifiable gold.

## Addendum Parts 2–3 (not executed here)

Part 2 (matched-generation factorial, E1–E4 estimands, C1 scoring, condition
M) and Part 3 (evidence-status rules) are design pre-registrations for
Study 2; no Study-2 data exist yet. The relevant Phase-0.5 feeders are in
place: E3's P3-replication baseline stands, C1's gold-U states must be
generated with the §2.4 prover-hygiene stamps (now implemented in the v1.1
metadata format), and the evidence-status list in §3 matches the erratum
and demotion applied here.

## Deliverables

```
analysis/phase0_5/
  report.md                       this file
  run.sh                          full reproduction (needs prover9/mace4)
  src/ladr.py                     FOL→LADR + prover runners + validation
  src/comparator_audit.py         §1.1
  src/transition_split.py         §1.2
  src/timeout_u_audit.py          §1.3
  src/repair_v1_1.py              §1.4
  tables/comparator_audit.csv, transition_split.csv, timeout_u_audit.csv
  figures/transition_split.{png,pdf}
  v1.1/{nl/, quarantine/, manifest.csv, CHANGELOG.md}
```
