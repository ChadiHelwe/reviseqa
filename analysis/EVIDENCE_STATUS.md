# Evidence Status — Study 1 (canonical, per SPEC_ADDENDUM_B)

Dated 2026-08-08, after Phase 0.6. This is the single source of truth for
what Study 1 does and does not support; the paper's claims must match it.
Every value below was re-verified against the machine-readable tables cited.
Study-1 analysis is **closed** as of Phase 0.6.

**Pre-registration of record for Study 2:** `SPEC_ADDENDUM_A.md` at commit
`d96341c193c2e5d3c8151f8db947e7d73d928d45` (short `d96341c19`). Its §1–2
(design, estimands E1–E4, C1 scoring, condition M, prover hygiene) are
unchanged by Addendum B.

## Not citable as findings (demotions)

| Claim | Status | Provenance |
|---|---|---|
| P1 DiD +15.5 [+9.0, +21.8] (suppression interaction) | **Comparator 18/18 invalid.** 11 inconsistent post-edit theories + 5 post-state mislabels (Phase 0.5 §1.1); the 2 survivors have prover-certified mislabeled *pre*-edit states — flips illusory (Phase 0.6 Job C). Appears in the paper only inside the data-quality narrative. | `phase0_5/tables/comparator_audit.csv`, `phase0_6/tables/c_leftover_comparator.csv` |
| P1b (pure-removal DiD) | **Not citable as evidence.** Audit-clean restriction: +11.6 [−6.6, +34.1], n.s., 5 unique turns; clean slice: incomputable (empty cells). Mention only as "no estimable effect," if at all. | `phase0_6/tables/a4_p1b_audit_clean.csv` |

## New structural fact (cite prominently)

On fully verified v1 data, **every answer-changing edit is a compound
remove+add flip** — zero pure-addition flips and zero pure-removal changed
turns survive verification. This is what FOL monotonicity requires; the
apparent compositional variety was metadata artifact. Consequence: **no
within-changed-band operation contrast is estimable from v1 at all** — the
operation-specificity hypothesis (visible retraction vs addition) has no v1
evidence in either direction.

**E1's status therefore changes** from "confirmatory replication of Phase-0
P1" to **"first estimable test of operation-specificity."** Gate thresholds
in Addendum A §2.2 unchanged. Paper language: *Study 1 characterizes the
failure mode; Study 2 tests its cause.*

## Citable Study-1 results (clean slice, certified examples)

1. **Flip autopsy — recommitment failure under visible retraction.**
   Retreat-to-Uncertain = 75.6% of flip-turn errors pooled; **85.5%
   explicit vs 50.1% implicit**; direction-symmetric (T→F 85.6% / F→T
   85.4%). Stale answers are the minority mode everywhere.
   (`phase0_6/tables/a1_flip_autopsy.csv`, `a5_transition_split.csv`)
2. **P3 answer-relevance null.** Operation cost ≈ 0 when answer-irrelevant:
   DiD −1.8 [−5.2, +1.7]. The failure is specific to answer-relevant
   retractions. (`phase0_6/tables/a2_p3_invariant.csv`)
3. **Invariant-band delta tax ≈ 30 pts** (implicit − explicit: add_only
   +31.4, removal +29.6), stable within ±2 pts of uncleaned estimates.
   (`phase0_6/tables/a3_invariant_descriptives.csv`)
4. **Composition-effect debunk** of the published 73.6 / 50.6 add/remove
   split: 73.6 reproduces exactly as recorded-add_only with harness scoring;
   50.6 only as *pure* removals (~1.7% of turns; 50.1 here);
   any-removal = 71.6%. (`phase0/tables/sanity_anchor.csv`)
5. **Demonstration-bug effect negligible (Job D):** anchoring fingerprint
   +0.7 pts; matched depression −0.34 pts at turns ≥ 3 (odd-flip strata
   unmatchable by parity; matching coverage 97–98%, stated per turn).
   → **301-clean absolute tables are reportable with this quantified
   caveat; no re-runs.** Job B preview (+4.6/+5.3/+5.9 at K=2/4/7) remains
   internal-only as the upper bound on published-number understatement.
   (`phase0_6/tables/d_verdict.csv`, `d1_*.csv`, `d2_*.csv`, `b_*.csv`)
6. **The forensics themselves:** four integrity findings (LaTeX-only prompt
   typo; demo-answer harness bug; `edits_made` ≠ actual context delta on
   18.2% of turns; stored prover input semantically ≠ context on 54 edits),
   the monotonicity argument, per-turn machine-checkable certificates, and
   the v1.1 release (728 clean / 32 flagged / 170 quarantined).
   (`phase0/report.md`, `phase0_5/report.md`, `phase0_5/v1.1/`)

## Design-motivation paragraph (for the paper, per Addendum B §4)

Monotonicity ⇒ answer-changing pure additions must be U→T and pure removals
T→U (or F→U) ⇒ pure-operation revision tests **require gold-Uncertain
states** ⇒ v1 contains exactly 2 such states (both initial, none post-edit;
`phase0_5/tables/timeout_u_audit.csv`) and, to our knowledge, no prior
logic benchmark contains verified ones ⇒ Study 2's B1/C1 are the first
verified pure-expansion / pure-contraction revision conditions, and the
Uncertain-attractor scoring problem (Addendum A §2.3) is intrinsic to this
territory, not an artifact of our design. This chain justifies both why the
question is unmeasured and why strict justification-gated scoring is
mandatory.

## Vocabulary (per Addendum A §3)

The phenomenon is **recommitment failure under visible retraction** —
models contract (abandon the old conclusion) but strand at the
non-committed midpoint instead of completing revision; rewritten context
releases the block; the effect is specific to answer-relevant retractions.
Avoid "suppression" (implies stale-belief persistence, the minority mode)
and bare "hedging" (P3 shows relevance-specificity).

## Unchanged

Study-2 design, estimands, gates, condition M, prover hygiene, timeline.
Remaining schedule risk is concentrated in Study-2 inference access.
