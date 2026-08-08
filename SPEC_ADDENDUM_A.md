# SPEC ADDENDUM A — Post-Phase-0 Revisions

Amends `reviseqa_control_spec.md` (matched generation) and `PHASE0_SPEC.md`.
Motivated by `analysis/phase0/report.md`. Three parts: (1) Phase 0.5
verification tasks for Claude Code, (2) revised confirmatory design and
scoring, (3) evidence-status rules for the paper.

---

## 1. Phase 0.5 — Verification follow-ups (Claude Code, ~half day)

Same hard rules as PHASE0_SPEC §1. Outputs under `analysis/phase0_5/`.

### 1.1 Comparator consistency audit (highest priority)
The 18 unique `add_only × changed` example-turns (and the 1 U→F) are the P1
comparator. By monotonicity of FOL, a T↔F flip achieved by pure addition
entails an **inconsistent post-edit theory** (Q remains provable; gold=F also
requires ¬Q provable) unless a prover resource limit mislabeled provability.
For each of these turns, and for the 48 pure-removal turns:

- Run Mace4 on the post-edit FOL context (finite model search, generous
  budget). Record: model found / no model within budget.
- Run Prover9 with target `$F` (derive a contradiction) as a cross-check.
- Re-derive gold: is Q provable? is ¬Q provable? at 10× the pipeline's
  original budget. Record proof/timeout status for each direction.
- Deliverable: `tables/comparator_audit.csv` + a verdict per turn:
  `CONSISTENT_GOLD_OK` / `INCONSISTENT` / `TIMEOUT_MISLABEL`.

**Decision rule:** if ≥ 1/3 of the 18 are `INCONSISTENT` or
`TIMEOUT_MISLABEL`, P1's DiD is demoted to "suggestive" everywhere (report
erratum note in `analysis/phase0/report.md` header; do not cite +15.5 as a
finding). P1b + autopsy + P3 remain Phase 0's citable results regardless.

### 1.2 Transition split (the undelivered §item)
Within `removal × changed`, split the explicit→implicit gain and the autopsy
(stale / uncertain_retreat) by transition T→F vs F→T. Deliverable: one table,
one figure. Prediction to test: retreat-to-Uncertain dominates both
directions (rules out "F is just hard to say").

### 1.3 Timeout-Uncertain audit
For every example in `verified-400` whose gold at any turn is U: confirm the
U was established by *completed* unprovability search in both directions,
not by timeout. If pipeline logs don't record this, re-run Prover9 both
directions at generous budget on all gold-U states. Report the rate of
`TIMEOUT_AS_U`. (Feeds §2.4: C1's gold-U must never be a timeout artifact.)

### 1.4 v1.1 repair script (extends the two queued bug fixes)
The repair deliverable now has four parts: (a) demo-answer fix
(`evaluation.py:214` → initial answer); (b) `edits_made` regenerated from
FOL context diffs for all 930 examples; (c) Mace4 consistency stamp on every
turn state; (d) prover-budget metadata (`proved` / `disproved` /
`exhausted` / `timeout` per direction per turn). Freeze as `v1.1` with
changelog. Any example failing (c) is quarantined, not shipped.

---

## 2. Matched-generation design revisions

### 2.1 Settings axis is now part of the factorial (confirmatory core)
Every changed-band condition (B1, C1, D1) is generated once and **presented
in both Explicit (delta) and Implicit (rewritten context) form** — same
items, two presentations. A/B0/C0 run explicit-primary with an implicit
subset (25%) to anchor the P3-replication.

### 2.2 Pre-registered confirmatory estimands (in order)
Let gain(X) = acc(X, implicit) − acc(X, explicit) on the first post-critical
probe, paired by seed.

- **E1 (primary, attractor-free):** gain(D1) − gain(B1).
  D1 gold=False, B1 gold=True — neither is the Uncertain attractor; both are
  single-critical-edit answer changes at matched proof depth. This is the
  clean replication of Phase-0 P1 with a valid comparator.
- **E2 (secondary, AGM-native):** gain(C1_strict) − gain(B1), C1 scored per
  §2.3.
- **E3 (background band):** gain(C0) − gain(B0) ≈ 0 predicted (P3
  replication). A non-null here reopens the surface-form explanation.
- **E4 (mirrored-pair asymmetry, within-explicit):** acc(B1) − acc(C1_strict)
  in the explicit setting on mirrored pairs (the original headline contrast,
  now demoted below E1 because of the attractor).

Gate for the ICLR claim: E1 ≥ 5 pts, CI excluding 0, direction-consistent in
≥ 60% of Tier-1 models, AND E3 CI containing 0. E2 concordant-in-direction
strengthens; E2 discordant triggers the §2.3 bias diagnostics before
interpretation.

### 2.3 C1 scoring protocol (attractor control)
Phase 0 showed retreat-to-Uncertain is the dominant error mode; C1's gold
*is* Uncertain, so lenient scoring rewards the failure mode.

- **Response format (all changed-band conditions):** answer ∈ {T,F,U} plus a
  one-line justification slot: for U, the model must name the retracted
  premise that previously carried the conclusion (or state that no
  derivation exists either way); for T/F, name the supporting premise(s).
- **C1_strict:** answer=U **and** justification identifies the removed
  kernel premise k (string-match against the kernel id list, fuzzy match
  threshold set on the pilot; borderline cases → LLM-judge with the kappa
  protocol's human ceiling).
- **C1_lenient:** answer=U regardless of justification. Report both;
  E2 uses strict.
- **Per-model Uncertain-bias covariate:** spurious-U rate on A/B0/C0 turns
  (gold≠U). Report C1_lenient − bias descriptively; include the bias term in
  the GLMM as a covariate. No algebraic "correction" as primary — strict
  scoring is the primary control, the covariate is the robustness.
- Justification slots are appended to *all* conditions identically so the
  format cost is constant across cells.

### 2.4 Prover hygiene (mandatory, was optional)
Mace4 satisfiability after every edit for every condition (pipeline
escapees are now demonstrated, not hypothetical). Gold-U requires completed
two-direction unprovability at generous budget; `exhausted` ≠ `timeout` is
recorded. All budgets logged in item metadata (§9 of the main spec).

### 2.5 Condition M — two-stage mechanism probe (30-seed subset)
On B1/C1/D1 items, a separate probe variant asks, after the critical edit:
(M1) "Does the previous answer's support still hold?" then (M2) the standard
query. Prediction from the recommitment story: M1 accuracy high (contraction
succeeds), M2 fails in explicit — the failure localizes between M1 and M2.
Run on 2 Tier-1 models only; never mixed into the main scoring runs (the
probe changes the task). One figure in the paper.

### 2.6 Unchanged
Mirrored-pair generation (§2 of main spec), matching table (§3), A/B0/C0
construction (§5–6), stratification and tiers (§7), thinking-ON/OFF
intervention, scaffold intervention, timeline. Sample-size note: E1 needs no
new conditions — D1 and B1 were already in the plan; the added cost is the
implicit-presentation twin of the changed band (~2× eval on ~43% of items)
plus the 25% implicit anchor for A/B0/C0.

---

## 3. Evidence-status rules for the paper

- **Study 1 (Phase 0, exploratory):** citable as findings — the autopsy
  asymmetry (84%/51% retreat), P3's answer-relevance null, the invariant-band
  descriptives, the composition-effect diagnosis of the published 73.6/50.6,
  and P1b as suggestive. **Not citable as a finding:** the P1 +15.5 DiD,
  pending §1.1; if the audit confirms corruption, it appears only inside the
  data-quality narrative.
- **Study 2 (matched, confirmatory):** E1–E4 pre-registered here, before any
  Study-2 data exist. The paper states this explicitly; it is the honest
  answer to "you amended your pre-registration in Study 1."
- **Erratum paragraph** (paper + repo): demo-answer bug, `edits_made`
  corruption + repair, harness scoring recoveries, LaTeX prompt typo, 930 vs
  933, comparator audit outcome. One place, stated plainly.
- Vocabulary: the phenomenon is **recommitment failure under visible
  retraction** — models contract (abandon the old conclusion) but strand at
  the non-committed midpoint instead of completing revision; rewritten
  context releases the block; the effect is specific to answer-relevant
  retractions. AGM anchor: internal revision truncated after contraction.
  Avoid "suppression" (implies stale-belief persistence, which is the
  minority mode) and avoid bare "hedging" (P3 shows relevance-specificity).
