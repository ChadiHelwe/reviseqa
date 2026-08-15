# SFT Data Reconstruction — Task Brief for Claude Code (private branch)

Purpose: transform the v1.1 clean pool into finetuning-ready data using
prefix expansion — each conversation of turn 0 + 7 edits yields the
prefixes {0}, {0,1}, {0,1,2}, …, {0..7}. Paper-2 asset; no training or
inference in this task. Outputs live on the private branch.

## 1. Hard rules
- Read-only inputs; all outputs under a NEW dedicated folder at repo
  root: `sft_data/v1/` (create it; add `sft_data/` to `.gitignore` —
  large generated JSONL should not be committed; the spec + `src/` are
  what's versioned). Seed 464; deterministic; no API calls; `run.sh` +
  pinned `requirements.txt`.
- Stop-and-ask conditions in §10 — never improvise past them.

## 2. Inputs

Repo root: `/reviseqa` — all paths below are
relative to it.

- **Authoritative source:** `analysis/phase0_5/v1.1/` — `manifest.csv`
  (the 728 clean example IDs and per-turn metadata: rederived_label,
  proof steps, consistency stamps) and `nl/` (repaired examples: fixed
  `edits_made`, with `edits_made_v1` preserved). If any repair field
  lives only in the manifest, join on example_id + edit_number.
- **Raw v1 (reference and cross-check only):**
  `reviseqa_data/nl/verified/` — full set, v1 format per §2.1; its
  `edits_made` and stored answers are untrusted. May be read to fill
  fields v1.1 left unchanged (contexts, `reasoning_chain`).
- **Test-split ID list:** `reviseqa_data/nl/verified-400/` (the Phase-0
  eval subset).

Precedence: build from v1.1; never source `edits_made` or labels from
raw.

### 2.1 Verified schema (against ex_12.json, v1 format)
Top level: `original_context` / `original_context_fol` (parallel lists),
`conclusion`, `conclusion_fol`, `answer` (canonical string, e.g. "True"),
`reasoning_chain` (list of steps, each {facts, rules, conclusion} with
{id, subject, text, fol, str_fol, negation}), `edits` (list of 7). Per
edit: `edit_number` (1-indexed), `modification_type` (e.g. INVARIANT —
record as metadata, never trust as edit_class; recode from FOL deltas),
`edited_context_fol` + `edited_natural_language_context` (parallel; the
implicit-setting source), `edits_made` {removed_facts, removed_rules,
added_facts, added_rules} of {fol, nl} (the explicit-setting source —
v1.1 repaired version only), `conclusion`(+`_fol`), `prover9_input`
(ignore), `answer` (v1 stored label — cross-check only), `model_results`
(per-judge {verified, mistake} for qwen3-235b-a22b-2507,
gemini-2.5-flash, gpt-5-mini — source for the a8 judge-agreement job).

**v1-format guard:** before building, verify loaded files carry v1.1
repair fields (`edits_made_v1` preserved alongside repaired `edits_made`,
and rederived labels available inline or via manifest join on
example_id + edit_number). If handed raw v1 files (like ex_12.json),
STOP — raw `edits_made` is corrupted on ~18% of turns.

## 3. Splits (fixed, emitted as manifests)
- **test** = clean ∩ eval-400 = 301 examples. Frozen; never used for
  training. Rationale: existing Phase-0 logs give item-matched baselines
  for every v1 model on exactly these items — zero-cost comparison.
- **train** = the remaining 427 clean examples, minus **dev** = 40
  examples sampled stratified by initial proof depth and flip count.
- Emit two alternative train/dev splits: (a) random-stratified;
  (b) shallow→deep (train on depth 6–7, dev/test analysis on 8–9) for the
  format-vs-capability comparison. Same frozen test set for both.

## 4. Reconstruction
For every example and every k in 0..7, define prefix P_k = turn 0 plus
edits 1..k. Build each prefix in BOTH settings:
- **explicit**: edits rendered as delta messages (removed facts / removed
  rules / added facts / added rules), using the *repaired* `edits_made`
  (actual FOL context deltas), never the v1 field.
- **implicit**: each edit turn presents the fully rewritten NL context.

Prompt templates: reuse the v1 evaluation templates (Appendix B) with the
harness fix applied — the turn-0 demonstrated answer is the turn-0
`rederived_label`, never the final edit's answer. Two prompt variants:
- **standard**: turn-0 answer shown, no reasoning.
- **cot0**: turn-0 answer plus the reasoning chain rendered
  deterministically from `reasoning_chain`: one sentence per step —
  "From {facts[*].text} and {rules[*].text}, it follows that
  {conclusion.text}." — joined in order; no LLM involved; edit turns get
  no reasoning (none exists — do not fabricate any).

Assistant target at every turn = the canonical label string of that turn's
`rederived_label`: exactly `True`, `False`, or `Uncertain`.

## 5. Output formats (both, from one builder)
1. **packed_{split}_{setting}_{variant}.jsonl** — one record per
   conversation: `messages` list with roles, plus `loss` flag per message
   (true only on assistant messages). Equivalent training signal to all
   prefixes at ~1/8 the tokens; preferred for axolotl/LLaMA-Factory-style
   trainers with input masking.
2. **prefix_{split}_{setting}_{variant}.jsonl** — one record per prefix:
   `prompt` (rendered through turn k's question), `completion` (label),
   `prefix_len` = k.

Per-record metadata (both formats): example_id, split, setting, variant,
per-turn {rederived_label, proof_steps, edit_class of the last edit,
answer_changed}, initial depth bin, flip count.

## 6. Stats report (`sft_data/v1/REPORT.md`) — mandatory before use
- Counts by split × setting × variant × prefix_len; token-length
  distributions (report tokenizer used).
- **Label distribution per turn position.** Known confound to surface
  explicitly: gold `Uncertain` is nearly absent in v1.1 (~2 initial
  states, none post-edit). Consequence: naive SFT can reduce the stall by
  teaching the model to never output Uncertain. State in the report that
  (a) stall-reduction claims require the ProverQA gold-U calibration
  check (U-precision/recall on ProverQA's Uncertain items) and (b) the
  training recipe must report Uncertain output rates before/after.
- Answer-trajectory shapes (flip counts, positions) by split.

## 7. Leakage check
Compute pairwise similarity between every train example and every test
example: Jaccard over the multiset of FOL predicates+constants of the
initial context. Report the max and the count of pairs > 0.8; list any
such pairs for manual review. (Shared first names alone are expected and
fine; near-duplicate premise sets are not.)

## 8. Unit tests (6 minimum)
- ex_12: expands to exactly 8 prefixes per setting; edit 1 is
  modification_type INVARIANT with 1 removed rule, 2 added facts, 1 added
  rule; the prefix-1 explicit prompt contains all four delta lines.
- The Uriel example expands to exactly 8 prefixes per setting; prefix 2's
  prompt contains edit 1 and edit 2 and not edit 3.
- Explicit prompts for a repaired-metadata example contain the removal
  that v1's `edits_made` omitted.
- Turn-0 demonstrated answer equals turn-0 rederived label on an example
  where v1's harness showed the wrong one.
- Packed loss flags: true exactly on assistant messages.
- Every completion ∈ {True, False, Uncertain} and matches rederived_label.

## 9. Deliverables
```
sft_data/v1/
  packed_*.jsonl  prefix_*.jsonl   (train/dev/test × explicit/implicit ×
                                    standard/cot0)
  manifests/split_random.csv  split_shallow_deep.csv
  REPORT.md  run.sh  src/  src/tests/
```

## 10. Stop-and-ask
`analysis/phase0_5/v1.1/` absent at the repo root (do NOT fall back to
raw); manifest or rederived labels missing; NL/FOL arrays misaligned for
any example; eval-400 ID list unrecoverable; any completion label absent;
leakage pairs > 0.8 exceed 5.
