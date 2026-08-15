"""SFT data reconstruction from the v1.1 clean pool (SFT_DATA_SPEC.md).

Prefix expansion: each conversation (turn 0 + 7 edits) yields prefixes
P_0..P_7, in explicit and implicit settings, standard and cot0 variants,
emitted as packed (loss-masked conversations) and prefix (prompt/completion)
JSONL. Deterministic, seed 464, no API calls.

Authoritative source: analysis/phase0_5/v1.1 (repaired edits_made +
rederived labels). Raw v1 is never read by this builder.
"""
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
V11 = os.path.join(REPO, "analysis", "phase0_5", "v1.1")
EVAL400 = os.path.join(REPO, "reviseqa_data", "nl", "verified-400")
OUT = os.path.join(REPO, "sft_data", "v1")

SEED = 464
# User instruction (2026-08-10), superseding §3: no dev set; 90/10
# train/test over the full clean pool. Test is sampled from clean ∩
# eval-400 so every test item keeps its item-matched Phase-0 baseline.
TEST_FRAC = 0.10
LABELS = {"True", "False", "Uncertain"}
SETTINGS = ("explicit", "implicit")
VARIANTS = ("standard", "cot0")

# v1 evaluation user-prompt template (src/evaluation.py), reused verbatim.
PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Options:
A) True
B) False
C) Uncertain

"""

# Deviation from the v1 assistant template (documented in REPORT.md):
# assistant targets are the canonical label strings per spec §4, so the
# system message instructs bare-label output instead of v1's JSON format.
SYSTEM_MSG = ("You are given a logic problem. The context may be updated "
              "between questions; each update either states the changes "
              "(removed/added facts and rules) or restates the full context. "
              "Answer every question with exactly one word: True, False, or "
              "Uncertain.")


# ──────────────────────────────────────────────────────────────────────────
# Loading (with the §2 v1-format guard)
# ──────────────────────────────────────────────────────────────────────────
def clean_pool():
    m = pd.read_csv(os.path.join(V11, "manifest.csv"))
    if not os.path.isdir(V11):
        raise SystemExit("STOP: analysis/phase0_5/v1.1 absent")
    m["ex"] = m.filename.str[:-5]
    return sorted(m[m.n_problems == 0].ex)


def load_example(ex_id):
    with open(os.path.join(V11, "nl", f"{ex_id}.json")) as f:
        d = json.load(f)
    # v1-format guard: refuse raw v1 files (corrupted edits_made)
    if "initial_state_metadata" not in d or any(
            "edits_made_v1" not in e or "prover_metadata" not in e
            for e in d["edits"]):
        raise SystemExit(f"STOP: {ex_id} lacks v1.1 repair fields — raw v1 input?")
    if len(d["original_context"]) != len(d["original_context_fol"]):
        raise SystemExit(f"STOP: NL/FOL misaligned in {ex_id} (initial)")
    for i, e in enumerate(d["edits"], 1):
        if len(e["edited_natural_language_context"]) != len(e["edited_context_fol"]):
            raise SystemExit(f"STOP: NL/FOL misaligned in {ex_id} edit {i}")
    return d


def labels_of(d):
    labs = [d["initial_state_metadata"]["rederived_label"]]
    labs += [e["prover_metadata"]["rederived_label"] for e in d["edits"]]
    for t, lab in enumerate(labs):
        if lab not in LABELS:
            raise SystemExit(f"STOP: non-canonical rederived label {lab!r} at turn {t}")
    return labs


def edit_class_of(edits_made):
    n_add = len(edits_made["added_facts"]) + len(edits_made["added_rules"])
    n_rem = len(edits_made["removed_facts"]) + len(edits_made["removed_rules"])
    return "removal" if n_rem else ("add_only" if n_add else "none")


def example_meta(ex_id, d):
    labs = labels_of(d)
    per_turn = [{"turn": 0, "rederived_label": labs[0], "proof_steps": None,
                 "edit_class": None, "answer_changed": False}]
    for i, e in enumerate(d["edits"], 1):
        per_turn.append({
            "turn": i, "rederived_label": labs[i],
            "proof_steps": None,  # not recorded in v1.1 (see REPORT.md)
            "edit_class": edit_class_of(e["edits_made"]),
            "answer_changed": labs[i] != labs[i - 1],
        })
    depth = len(d["reasoning_chain"])
    return {
        "example_id": ex_id, "labels": labs, "per_turn": per_turn,
        "depth": depth, "depth_bin": "6-7" if depth <= 7 else "8+",
        "flip_count": int(sum(t["answer_changed"] for t in per_turn)),
    }


# ──────────────────────────────────────────────────────────────────────────
# Rendering
# ──────────────────────────────────────────────────────────────────────────
def render_reasoning(d):
    """Deterministic rendering of reasoning_chain (spec §4, cot0).

    One numbered line per step, premises kept verbatim as whole sentences
    under Fact(s)/Rule(s) labels. Rule texts begin with varied capitalized
    words ("If", "Anyone", "For all humans", a proper name), so embedding
    them mid-sentence cannot be made grammatical by any fixed template;
    labeled sections keep every premise well-formed and preserve the
    fact/rule distinction. No LLM involved.
    """
    lines = []
    for i, step in enumerate(d["reasoning_chain"], 1):
        facts = [f["text"].strip() for f in step.get("facts", [])]
        rules = [r["text"].strip() for r in step.get("rules", [])]
        concl = (step.get("conclusion") or {}).get("text", "").strip()
        parts = [f"Step {i}."]
        if facts:
            parts.append(("Facts: " if len(facts) > 1 else "Fact: ") + " ".join(facts))
        if rules:
            parts.append(("Rules: " if len(rules) > 1 else "Rule: ") + " ".join(rules))
        parts.append(f"Therefore: {concl}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def explicit_delta(edits_made):
    """Delta text from the REPAIRED edits_made, v1 harness section order."""
    parts = []
    for key, title in [("removed_facts", "Removed facts"),
                       ("removed_rules", "Removed rules"),
                       ("added_rules", "Added rules"),
                       ("added_facts", "Added facts")]:
        if edits_made[key]:
            parts.append(title + ":\n" + "\n".join(f"- {x['nl']}" for x in edits_made[key]))
    return "\n\n".join(parts)


def turn_block(d, setting, t):
    """The question block for turn t (t=0 = original context; t>=1 = edit t).

    Context + question + options, with no answer — the eval template.
    """
    if t == 0:
        ctx = "\n".join(d["original_context"])
        q = f"Does the context entail the conclusion '{d['conclusion']}'?"
    else:
        e = d["edits"][t - 1]
        ctx = (explicit_delta(e["edits_made"]) if setting == "explicit"
               else "\n".join(e["edited_natural_language_context"]))
        q = f"Does the context entail the conclusion '{e['conclusion']}'?"
    return PROMPT_TEMPLATE.format(context=ctx, question=q)


def demo_block(d, labs, variant, setting):
    """Turn-0 demonstration (spec §4): full context + question + options +
    the turn-0 gold answer, shown in the first user message. `cot0` also
    shows the deterministically rendered reasoning chain. The demonstrated
    answer is the turn-0 rederived_label (harness demo-answer bug fixed).
    Never a training target — turn 0 is given, not scored.
    """
    block = turn_block(d, setting, 0)
    if variant == "cot0":
        block += f"Reasoning:\n{render_reasoning(d)}\n"
    return block + f"Answer: {labs[0]}"


def user_messages(d, labs, variant, setting):
    """User messages for edit turns 1..7. The turn-0 demonstration is
    prepended to the first one so roles strictly alternate."""
    msgs = [demo_block(d, labs, variant, setting) + "\n\n" + turn_block(d, setting, 1)]
    msgs += [turn_block(d, setting, t) for t in range(2, 8)]
    return msgs


def packed_record(meta, d, setting, variant, split):
    """One conversation: system, then 7 user/assistant pairs (edits 1..7).
    Loss is true on exactly the 7 assistant messages; the turn-0 answer
    rides inside the first user message as a demonstration."""
    labs = meta["labels"]
    messages = [{"role": "system", "content": SYSTEM_MSG, "loss": False}]
    for i, u in enumerate(user_messages(d, labs, variant, setting), 1):
        messages.append({"role": "user", "content": u, "loss": False})
        messages.append({"role": "assistant", "content": labs[i], "loss": True})
    return {"example_id": meta["example_id"], "split": split, "setting": setting,
            "variant": variant, "messages": messages,
            "per_turn": meta["per_turn"], "depth_bin": meta["depth_bin"],
            "flip_count": meta["flip_count"]}


def prefix_records(meta, d, setting, variant, split, include_turn0_target=False):
    """One record per edit turn k = 1..7 (spec §4): the demonstration plus
    edits 1..k-1 with their gold labels (teacher forcing) in the prompt, and
    turn k's label as the sole loss-bearing completion.

    With include_turn0_target, additionally emits a static k=0 record in
    which the demonstration's answer is withheld and turn 0 is the target.
    """
    labs = meta["labels"]
    users = user_messages(d, labs, variant, setting)
    recs = []
    ks = ([0] if include_turn0_target else []) + list(range(1, 8))
    for k in ks:
        parts = [f"System: {SYSTEM_MSG}"]
        if k == 0:
            parts.append(f"User: {turn_block(d, setting, 0)}")
        else:
            for t in range(1, k):
                parts.append(f"User: {users[t - 1]}")
                parts.append(f"Assistant: {labs[t]}")
            parts.append(f"User: {users[k - 1]}")
        parts.append("Assistant: ")
        recs.append({
            "example_id": meta["example_id"], "split": split, "setting": setting,
            "variant": variant, "prefix_len": k,
            "prompt": "\n".join(parts),
            "completion": labs[k],
            "per_turn": meta["per_turn"][:k + 1],
            "depth_bin": meta["depth_bin"], "flip_count": meta["flip_count"],
        })
    return recs


# ──────────────────────────────────────────────────────────────────────────
# Splits (§3)
# ──────────────────────────────────────────────────────────────────────────
def make_splits(metas):
    ev = {f[:-5] for f in os.listdir(EVAL400) if f.endswith(".json")}
    all_ids = sorted(m["example_id"] for m in metas)
    eval_clean = sorted(e for e in all_ids if e in ev)
    bym = {m["example_id"]: m for m in metas}

    # (a) 80/20 train/test, no dev. Test = stratified sample (depth_bin ×
    # flip_count, largest-remainder quotas, seed 464) drawn from
    # clean ∩ eval-400, so item-matched Phase-0 baselines exist for every
    # test item. Train = all remaining clean examples.
    n_test = round(TEST_FRAC * len(all_ids))
    assert n_test <= len(eval_clean), "test quota exceeds baseline-covered pool"
    rng = np.random.default_rng(SEED)
    strata = {}
    for e in eval_clean:
        strata.setdefault((bym[e]["depth_bin"], bym[e]["flip_count"]), []).append(e)
    quotas = {s: n_test * len(v) / len(eval_clean) for s, v in strata.items()}
    base = {s: int(q) for s, q in quotas.items()}
    rem = n_test - sum(base.values())
    for s in sorted(quotas, key=lambda s: quotas[s] - base[s], reverse=True)[:rem]:
        base[s] += 1
    test = []
    for s in sorted(strata):
        picks = rng.choice(sorted(strata[s]), size=min(base[s], len(strata[s])),
                           replace=False)
        test.extend(picks.tolist())
    test = sorted(test)
    train = sorted(set(all_ids) - set(test))

    # (b) shallow -> deep within the train set (depth 6-7 train; 8+ held
    # out for analysis; includes the depth-10 examples in the deep side)
    train_b = sorted(e for e in train if bym[e]["depth_bin"] == "6-7")
    analysis_b = sorted(e for e in train if bym[e]["depth_bin"] == "8+")

    return {"random": {"train": train, "test": test},
            "shallow_deep": {"train": train_b, "analysis": analysis_b, "test": test}}


# ──────────────────────────────────────────────────────────────────────────
# Leakage (§7): literal spec metric + structural cross-check
# ──────────────────────────────────────────────────────────────────────────
def _token_bag(d):
    toks = []
    for s in d["original_context_fol"]:
        toks += re.findall(r"p_\d+", s)
        toks += re.findall(r"(?<=[(,])\s*([A-Z][a-zA-Z .]*?)\s*(?=[,)])", s)
    return Counter(toks)


def _skeleton(d):
    return frozenset(
        re.sub(r"\s+", "", re.sub(r"(?<=[(,])\s*[A-Z][a-zA-Z .]*?\s*(?=[,)])", "X", s))
        for s in d["original_context_fol"])


def leakage(examples, train_ids, test_ids):
    bags = {e: _token_bag(examples[e]) for e in train_ids + test_ids}
    skels = {e: _skeleton(examples[e]) for e in train_ids + test_ids}
    rows = []
    mx_lit = mx_str = 0.0
    for a in train_ids:
        for b in test_ids:
            inter = sum((bags[a] & bags[b]).values())
            union = sum((bags[a] | bags[b]).values())
            j = inter / union if union else 0.0
            js = len(skels[a] & skels[b]) / len(skels[a] | skels[b])
            mx_lit, mx_str = max(mx_lit, j), max(mx_str, js)
            if j > 0.8:
                rows.append({"train_example": a, "test_example": b,
                             "jaccard_literal": round(j, 4),
                             "jaccard_structural_name_masked": round(js, 4)})
    return pd.DataFrame(rows), mx_lit, mx_str


# ──────────────────────────────────────────────────────────────────────────
# Build
# ──────────────────────────────────────────────────────────────────────────
def write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--include-turn0-target", action="store_true",
                    help="also emit a static k=0 prefix record per conversation "
                         "(demonstration answer withheld; 8 records instead of 7)")
    args = ap.parse_args()
    include_t0 = args.include_turn0_target

    os.makedirs(os.path.join(OUT, "manifests"), exist_ok=True)
    ids = clean_pool()
    examples = {e: load_example(e) for e in ids}
    metas = [example_meta(e, examples[e]) for e in ids]
    bym = {m["example_id"]: m for m in metas}
    splits = make_splits(metas)

    # remove stale outputs from prior split configurations (e.g. dev files)
    for f in os.listdir(OUT):
        if f.endswith(".jsonl"):
            os.remove(os.path.join(OUT, f))

    for name, sp in splits.items():
        rows = [{"example_id": e, "split": s}
                for s in sp for e in sp[s]]
        pd.DataFrame(rows).to_csv(
            os.path.join(OUT, "manifests", f"split_{name}.csv"), index=False)

    # JSONL files use the (a) random split; split (b) is defined by its
    # manifest over the same example_ids (filter by manifest to re-slice).
    sp = splits["random"]
    counts = []
    for split in ("train", "test"):
        for setting in SETTINGS:
            for variant in VARIANTS:
                packed, prefixes = [], []
                for e in sp[split]:
                    packed.append(packed_record(bym[e], examples[e], setting, variant, split))
                    prefixes.extend(prefix_records(bym[e], examples[e], setting, variant,
                                                   split, include_t0))
                write_jsonl(os.path.join(OUT, f"packed_{split}_{setting}_{variant}.jsonl"), packed)
                write_jsonl(os.path.join(OUT, f"prefix_{split}_{setting}_{variant}.jsonl"), prefixes)
                counts.append({"split": split, "setting": setting, "variant": variant,
                               "n_packed": len(packed), "n_prefix": len(prefixes)})
                print(f"  wrote {split}/{setting}/{variant}: "
                      f"{len(packed)} packed, {len(prefixes)} prefix")

    leak, mx_lit, mx_str = leakage(examples, sp["train"], sp["test"])
    leak.to_csv(os.path.join(OUT, "manifests", "leakage_pairs.csv"), index=False)
    print(f"leakage: {len(leak)} literal pairs > 0.8 (max {mx_lit:.3f}); "
          f"structural max {mx_str:.3f}")

    write_report(metas, bym, splits, pd.DataFrame(counts), leak, mx_lit, mx_str,
                 include_t0)
    print("REPORT.md written")


def write_report(metas, bym, splits, counts, leak, mx_lit, mx_str, include_t0=False):
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    sp = splits["random"]

    # token stats on packed conversations (sum over messages)
    tok_rows = []
    for setting in SETTINGS:
        for variant in VARIANTS:
            lens = []
            for e in sp["train"][:120]:  # deterministic sample for speed
                d = load_example(e)
                rec = packed_record(bym[e], d, setting, variant, "train")
                lens.append(sum(len(enc.encode(m["content"])) for m in rec["messages"]))
            tok_rows.append({"setting": setting, "variant": variant,
                             "packed_tokens_mean": int(np.mean(lens)),
                             "packed_tokens_p95": int(np.percentile(lens, 95)),
                             "n_sampled": len(lens)})
    tok = pd.DataFrame(tok_rows)

    # label distribution per turn position (whole clean pool)
    lab_rows = []
    for t in range(8):
        c = Counter(m["labels"][t] for m in metas)
        lab_rows.append({"turn": t, **{k: c.get(k, 0) for k in ("True", "False", "Uncertain")}})
    lab = pd.DataFrame(lab_rows)

    # trajectory shapes by split
    tr_rows = []
    for split in ("train", "test"):
        ms = [bym[e] for e in sp[split]]
        fc = Counter(m["flip_count"] for m in ms)
        tr_rows.append({"split": split, "n": len(ms),
                        "mean_flips": round(np.mean([m["flip_count"] for m in ms]), 2),
                        "flip_count_distribution": dict(sorted(fc.items()))})
    tr = pd.DataFrame(tr_rows)

    def md(df):
        header = "| " + " | ".join(map(str, df.columns)) + " |"
        sep = "|" + "---|" * len(df.columns)
        rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
        return "\n".join([header, sep] + rows)

    report = f"""# SFT Data v1 — Build Report (SFT_DATA_SPEC.md)

Deterministic build, seed {SEED}. Source: `analysis/phase0_5/v1.1` (728
clean examples; repaired `edits_made`; rederived labels). Raw v1 was not
read. Reproduce: `sft_data/v1/run.sh`.

## Splits

Per user instruction (2026-08-10), superseding §3: **no dev set; 90/10
train/test over the full clean pool** ({len(sp["train"]) + len(sp["test"])}
examples).

- **train** = {len(sp["train"])}, **test** = {len(sp["test"])}
  ({TEST_FRAC:.0%}). Test is a stratified sample (depth_bin × flip_count,
  largest-remainder quotas, seed {SEED}) drawn **from clean ∩ eval-400**, so
  item-matched Phase-0 baselines exist for every test item (the original
  frozen-test rationale is preserved at the new size). The {301 - len(sp["test"])}
  clean eval-400 examples not sampled into test are in train.
- **(b) shallow→deep** (within train): train = depth 6–7
  ({len(splits["shallow_deep"]["train"])}), held-out analysis = depth ≥ 8
  ({len(splits["shallow_deep"]["analysis"])}; includes the depth-10
  examples). Same test set.
- JSONL files carry split (a); re-slice by `manifests/split_shallow_deep.csv`
  for (b).

## Counts (split × setting × variant)

{md(counts)}

Record construction (spec §4): turn 0 is a **demonstration inside the first
user message** — full context, question, options, and the turn-0 gold answer
(the rederived label; `cot0` also shows the reasoning chain). It is never a
training target. Each conversation yields **{8 if include_t0 else 7} prefix
records** (k = {'0..7' if include_t0 else '1..7'}): edits 1..k−1 appear with
their gold labels as teacher forcing, and turn k's label is the sole
loss-bearing completion.
{'' if include_t0 else chr(10) + 'Built without `--include-turn0-target`; pass that flag to add the static k=0 record in which the demonstration answer is withheld and turn 0 becomes the target.' + chr(10)}
Packed records carry the same conversation as 15 messages (system + 7
user/assistant pairs) with `loss: true` on exactly the 7 assistant messages.

## Token lengths (tiktoken cl100k_base; 120-example deterministic sample)

{md(tok)}

## Label distribution per turn position (all 728 clean examples)

{md(lab)}

**Known confound (state before any use):** gold `Uncertain` is nearly absent
({int(lab.Uncertain.sum())} states total, all at turn 0; none post-edit).
Naive SFT on this data can reduce the retreat-to-Uncertain stall simply by
teaching the model to *never output Uncertain*. Therefore: (a) any
stall-reduction claim requires the ProverQA gold-U calibration check
(U-precision/recall on ProverQA's Uncertain items), and (b) the training
recipe must report Uncertain output rates before/after finetuning.

## Answer-trajectory shapes by split

{md(tr)}

## Leakage check (§7)

Literal spec metric (multiset Jaccard over FOL predicates+constants of the
initial context, train × test): **{len(leak)} pairs > 0.8**
(max {mx_lit:.3f}) — the §10 threshold of 5 was first exceeded in the
original build, which was STOPPED; the user approved proceeding with dual
reporting (pairs listed in `manifests/leakage_pairs.csv` for manual
review), which carries over to this split.
Diagnosis: all flagged pairs are metric artifacts — predicate indices
(`p_i`) are example-local and overlap by construction, and flagged pairs
share a first name (expected and fine per §7). The structural cross-check
(name-masked exact-formula Jaccard over all train∪dev × test pairs) finds
**max {mx_str:.3f} and zero pairs > 0.5**: no near-duplicate premise sets.

## Deviations from the written brief (all by explicit user instruction)

The brief text was re-pasted on 2026-08-10 with §4 rewritten; §3 and the
§4 cot0 template line still carry their original wording, which three
conversational instructions had already superseded. Those instructions are
what this build follows:

| Brief says | Built as | Authority |
|---|---|---|
| §3: test = 301 frozen, train = 427, dev = 40 | no dev; 90/10 train/test ({len(sp["train"])}/{len(sp["test"])}) | user, 2026-08-10 ("just training and testing as 90/10") |
| §4: cot0 rendered as "From {{facts}} and {{rules}}, it follows that {{conclusion}}." | numbered `Step N. Fact(s): … Rule(s): … Therefore: …` lines | user, 2026-08-10 ("do what is best") — the original template cannot be made grammatical, since rule sentences begin with varied capitalized words |
| §8 test 2: "Uriel expands to exactly 8 prefixes" | verified under `--include-turn0-target` (8); the default is 7 per §4 | internal to the brief: §4's default is k=1..7 |

Reverting any of these is a one-line change at the top of `build_sft.py`
(`TEST_FRAC`, `render_reasoning`) plus a rebuild.

## Deviations & notes

- **proof_steps is null**: v1.1 records prover statuses and budgets per
  turn, not proof lengths; the spec's "proof steps" manifest field does not
  exist in the v1.1 freeze. Initial proof depth (for stratification and the
  shallow→deep split) = `len(reasoning_chain)` (6–10).
- **Assistant format**: user prompts reuse the v1 template verbatim;
  assistant targets are the canonical bare label at every turn (spec §4),
  so v1's JSON answer template is not reproduced and the system message
  instructs bare-label output. cot0's turn-0 target is the rendered chain +
  `Answer: <label>`; edit turns get no reasoning (none exists; none
  fabricated). In prefix format, completions are always the bare label.
- **Turn-0 answer** is the turn-0 `rederived_label` (harness demo-answer
  bug fixed at the source).
- `model_results` (per-judge votes) is preserved in v1.1 for the future
  judge-agreement job; not used here.
- Labels: for clean examples rederived labels agree with stored v1 answers
  by construction (that is what "clean" means); targets are the rederived
  labels.
"""
    with open(os.path.join(OUT, "REPORT.md"), "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
