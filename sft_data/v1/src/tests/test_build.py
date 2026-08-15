"""SFT builder unit tests (SFT_DATA_SPEC.md §8 — 6 required tests).

Run: python sft_data/v1/src/tests/test_build.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from build_sft import (  # noqa: E402
    LABELS, example_meta, load_example, packed_record, prefix_records,
)


def _prefixes(ex_id, setting="explicit", variant="standard"):
    d = load_example(ex_id)
    meta = example_meta(ex_id, d)
    return d, meta, prefix_records(meta, d, setting, variant, "train")


def test_ex12_expansion_teacher_forcing_and_edit1_composition():
    # ex_12: exactly 7 records per setting by default, 8 with the flag; the
    # k=5 record carries gold assistant answers for edits 1-4 only and its
    # completion is edit 5's label; repaired edit 1 = INVARIANT with
    # 1 removed rule, 2 added facts, 1 added rule, and the k=1 explicit
    # prompt carries all four delta lines.
    d, meta, recs = _prefixes("ex_12")
    assert len(recs) == 7 and [r["prefix_len"] for r in recs] == list(range(1, 8))
    _, _, recs_impl = _prefixes("ex_12", setting="implicit")
    assert len(recs_impl) == 7
    with_t0 = prefix_records(meta, d, "explicit", "standard", "train",
                             include_turn0_target=True)
    assert len(with_t0) == 8 and with_t0[0]["prefix_len"] == 0

    r5 = next(r for r in recs if r["prefix_len"] == 5)
    filled = [ln for ln in r5["prompt"].split("\n") if ln.startswith("Assistant: ")]
    assert [ln[len("Assistant: "):] for ln in filled] == meta["labels"][1:5] + [""]
    assert r5["completion"] == meta["labels"][5]
    assert r5["prompt"].endswith("Assistant: ")

    e1 = d["edits"][0]
    assert e1["modification_type"] == "INVARIANT"
    em = e1["edits_made"]
    assert (len(em["removed_rules"]), len(em["added_facts"]),
            len(em["added_rules"]), len(em["removed_facts"])) == (1, 2, 1, 0)
    p1 = next(r for r in recs if r["prefix_len"] == 1)["prompt"]
    assert "Removed rules:" in p1 and "Added facts:" in p1 and "Added rules:" in p1
    for section in ("removed_rules", "added_facts", "added_rules"):
        for item in em[section]:
            assert f"- {item['nl']}" in p1, (section, item["nl"])


def test_uriel_prefix2_contains_edits_1_2_not_3():
    # Uriel = ex_2512: 8 prefixes per setting under --include-turn0-target;
    # prefix-2 prompt has edit 1 and edit 2 content but not edit 3's.
    d = load_example("ex_2512")
    meta = example_meta("ex_2512", d)
    recs = prefix_records(meta, d, "explicit", "standard", "train",
                          include_turn0_target=True)
    assert len(recs) == 8
    assert len(prefix_records(meta, d, "implicit", "standard", "train",
                              include_turn0_target=True)) == 8
    p2 = next(r for r in recs if r["prefix_len"] == 2)["prompt"]
    e1_nl = d["edits"][0]["edits_made"]["added_facts"][0]["nl"]
    e2 = d["edits"][1]["edits_made"]
    e2_nl = (e2["removed_rules"] + e2["added_rules"] + e2["added_facts"]
             + e2["removed_facts"])[0]["nl"]
    e3 = d["edits"][2]["edits_made"]
    e3_items = e3["removed_rules"] + e3["added_rules"] + e3["added_facts"] + e3["removed_facts"]
    assert f"- {e1_nl}" in p2 and f"- {e2_nl}" in p2
    assert all(f"- {x['nl']}" not in p2 for x in e3_items)


def test_repaired_metadata_removal_present():
    # ex_1011 edit 6: v1's edits_made omitted the removed biconditional;
    # the repaired field (and hence the explicit prompt) must contain it.
    d, meta, recs = _prefixes("ex_1011")
    e6 = d["edits"][5]
    v1_removed = {x["nl"] for k in ("removed_facts", "removed_rules")
                  for x in e6["edits_made_v1"][k]}
    rep_removed = {x["nl"] for k in ("removed_facts", "removed_rules")
                   for x in e6["edits_made"][k]}
    gained = rep_removed - v1_removed
    assert gained, "expected a removal that v1 omitted"
    p6 = next(r for r in recs if r["prefix_len"] == 6)["prompt"]
    assert any(f"- {nl}" in p6 for nl in gained)


def test_turn0_demonstrated_answer_is_rederived_label():
    # ex_1003: initial answer != final edit answer, so the buggy v1 harness
    # demonstrated the wrong turn-0 answer. The demonstration in our first
    # user message must show the turn-0 rederived label.
    d = load_example("ex_1003")
    meta = example_meta("ex_1003", d)
    assert d["answer"] != d["edits"][-1]["answer"]
    rederived = d["initial_state_metadata"]["rederived_label"]
    rec = packed_record(meta, d, "explicit", "standard", "train")
    first_user = rec["messages"][1]
    assert first_user["role"] == "user" and first_user["loss"] is False
    demo = first_user["content"].split("Context:", 2)[1]  # turn-0 block only
    assert f"Answer: {rederived}" in demo
    assert f"Answer: {d['edits'][-1]['answer']}" not in demo
    # turn 0 is never a target: assistant messages are exactly turns 1..7
    targets = [m["content"] for m in rec["messages"] if m["role"] == "assistant"]
    assert targets == meta["labels"][1:8]


def test_packed_loss_flags():
    d = load_example("ex_12")
    meta = example_meta("ex_12", d)
    for variant in ("standard", "cot0"):
        rec = packed_record(meta, d, "implicit", variant, "train")
        for msg in rec["messages"]:
            assert msg["loss"] == (msg["role"] == "assistant"), msg["role"]
        assert sum(m["loss"] for m in rec["messages"]) == 7   # edits 1..7
        assert len(rec["messages"]) == 15                     # system + 7 pairs


def test_cot0_rendering_wellformed():
    # Every step line names its conclusion and keeps premises verbatim as
    # whole sentences; the target ends with the turn-0 label.
    from build_sft import render_reasoning
    d = load_example("ex_12")
    meta = example_meta("ex_12", d)
    text = render_reasoning(d)
    lines = text.split("\n")
    assert len(lines) == len(d["reasoning_chain"])
    for i, (line, step) in enumerate(zip(lines, d["reasoning_chain"]), 1):
        assert line.startswith(f"Step {i}. ")
        for f in step.get("facts", []):
            assert f["text"].strip() in line
        for r in step.get("rules", []):
            assert r["text"].strip() in line
        assert line.endswith("Therefore: " + step["conclusion"]["text"].strip())
        assert ("Rule: " in line or "Rules: " in line) == bool(step.get("rules"))
    rec = packed_record(meta, d, "explicit", "cot0", "train")
    demo = rec["messages"][1]["content"]
    assert f"Reasoning:\n{text}\nAnswer: {meta['labels'][0]}" in demo


def test_completions_canonical_and_match_rederived():
    for ex_id in ("ex_12", "ex_2512", "ex_1003"):
        d = load_example(ex_id)
        meta = example_meta(ex_id, d)
        for setting in ("explicit", "implicit"):
            for variant in ("standard", "cot0"):
                for flag in (False, True):
                    recs = prefix_records(meta, d, setting, variant, "train",
                                          include_turn0_target=flag)
                    for r in recs:
                        assert r["completion"] in LABELS
                        assert r["completion"] == meta["labels"][r["prefix_len"]]
                        assert r["per_turn"][-1]["rederived_label"] == r["completion"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
