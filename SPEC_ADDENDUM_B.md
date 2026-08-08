# SPEC ADDENDUM B — Evidence-Status Update after Phase 0.6

Dated 2026-08-08, after `analysis/phase0_6/report.md`. Amends **only** §3
(evidence-status rules) of SPEC_ADDENDUM_A. A's §1–2 — design, estimands
E1–E4, C1 scoring, condition M — are unchanged and A remains the
pre-registration of record for Study 2 (cite its commit hash in the paper).

## 1. Demotions

- **P1 comparator: 18/18 invalid** (Phase 0.5 §1.1: 16 inconsistent or
  post-state-mislabeled; Phase 0.6 Job C: the 2 survivors have mislabeled
  *pre*-edit states, flips illusory). The +15.5 DiD appears in the paper
  only inside the data-quality narrative, never as a finding.
- **P1b: not citable as evidence.** Audit-clean restriction: +11.6
  [−6.6, +34.1], n.s., 5 unique turns; clean slice: incomputable (empty
  cells). Mention only as "no estimable effect," if at all.

## 2. New structural fact (cite prominently)

On fully verified v1 data, **every answer-changing edit is a compound
remove+add flip** — zero pure-addition flips, zero pure-removal changed
turns survive. This is what FOL monotonicity requires; the apparent
compositional variety was metadata artifact. Consequence: **no
within-changed-band operation contrast is estimable from v1 at all.** The
operation-specificity hypothesis (visible retraction vs addition) has no v1
evidence in either direction.

- **E1's status changes: from "confirmatory replication of Phase-0 P1" to
  "first estimable test of operation-specificity."** Gate thresholds in A
  §2.2 unchanged. Paper language: Study 1 characterizes the failure mode;
  Study 2 tests its cause.

## 3. Confirmed citable Study-1 results (clean slice, certified examples)

1. Flip autopsy: retreat-to-Uncertain 75.6% of errors pooled; 85.5%
   explicit vs 50.1% implicit; direction-symmetric (T→F 85.6 / F→T 85.4).
2. P3 answer-relevance null: −1.8 [−5.2, +1.7] (operation cost ≈ 0 when
   answer-irrelevant).
3. Invariant-band delta tax ≈ 30 pts, stable within ±2 pts of the
   uncleaned estimates.
4. Composition-effect debunk of the published 73.6/50.6 split.
5. Job D: demonstration-bug effect negligible (anchoring +0.7 pts;
   matched depression −0.34 pts at turns ≥ 3; odd-flip strata unmatchable
   by parity — state coverage) → **301-clean absolute tables are reportable
   with this quantified caveat; no re-runs**. Job B preview (+4.6/+5.3/+5.9
   at K=2/4/7) remains internal-only as the upper bound on published-number
   understatement.
6. The forensics themselves: four integrity findings, the monotonicity
   argument, per-turn machine-checkable certificates, v1.1.

## 4. Design-motivation paragraph for the paper (new)

Monotonicity ⇒ answer-changing pure additions must be U→T and pure removals
T→U (or F→U) ⇒ pure-operation revision tests **require gold-Uncertain
states** ⇒ v1 contains 2 such states (both initial, none post-edit) and, to
our knowledge, no prior logic benchmark contains verified ones ⇒ Study 2's
B1/C1 are the first verified pure-expansion / pure-contraction revision
conditions, and the Uncertain-attractor scoring problem (A §2.3) is
intrinsic to this territory, not an artifact of our design. This chain
justifies both why the question is unmeasured and why strict justification-
gated scoring is mandatory.

## 5. Unchanged

Study-2 design, estimands, gates, condition M, prover hygiene, timeline.
All remaining schedule risk is concentrated in Study-2 inference access
(zero-budget scramble per plan); Study-1 analysis is closed as of Phase 0.6.
