"""Coder unit tests: 5 hand-checked cases from the dataset (PHASE0_SPEC §3)
plus answer-normalization checks.

Run: python analysis/phase0/src/tests/test_coder.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
DATA = os.path.join(REPO, "reviseqa_data", "nl", "verified-400")

from code_edits import classify_edit, edit_target, normalize_pred, transition  # noqa: E402


def load(ex_id):
    with open(os.path.join(DATA, f"{ex_id}.json")) as f:
        return json.load(f)


def test_uriel_edit1_add_only_invariant():
    # Paper's Uriel example = ex_2512. Edit #1: 3 added facts, nothing removed.
    d = load("ex_2512")
    e = d["edits"][0]
    assert classify_edit(e["edits_made"]) == "add_only"
    assert e["answer"] == d["answer"] == "False"          # invariant
    assert transition(d["answer"], e["answer"]) == "F→F"
    assert edit_target(e["edits_made"]) == "fact"


def test_uriel_edit2_removal_changed_rule():
    # Edit #2: 1 rule removed + 1 rule added -> removal, flip False->True.
    d = load("ex_2512")
    e = d["edits"][1]
    assert classify_edit(e["edits_made"]) == "removal"
    assert (d["edits"][0]["answer"], e["answer"]) == ("False", "True")  # changed
    assert transition(d["edits"][0]["answer"], e["answer"]) == "F→T"
    assert edit_target(e["edits_made"]) == "rule"


def test_ex1006_edit1_removal_changed_fact():
    # 1 fact removed + 2 facts added -> removal; True -> False flip; fact-only.
    d = load("ex_1006")
    e = d["edits"][0]
    assert classify_edit(e["edits_made"]) == "removal"
    assert (d["answer"], e["answer"]) == ("True", "False")
    assert transition(d["answer"], e["answer"]) == "T→F"
    assert edit_target(e["edits_made"]) == "fact"


def test_ex1006_edit2_removal_changed_both():
    # 2 facts removed + 1 rule added -> removal; False -> True; target both.
    d = load("ex_1006")
    e = d["edits"][1]
    assert classify_edit(e["edits_made"]) == "removal"
    assert (d["edits"][0]["answer"], e["answer"]) == ("False", "True")
    assert edit_target(e["edits_made"]) == "both"


def test_prev_ex352_edit1_none_class():
    # Dataset artifact: empty edits_made but answer changes U->T.
    d = load("prev_ex_352")
    e = d["edits"][0]
    assert classify_edit(e["edits_made"]) == "none"
    assert transition(d["answer"], e["answer"]) == "U→T"
    assert edit_target(e["edits_made"]) == "none"


def test_normalize_pred():
    assert normalize_pred("True") == "True"
    assert normalize_pred("false") == "False"
    assert normalize_pred("Uncertain") == "Uncertain"
    assert normalize_pred("A") == "True"
    assert normalize_pred("B") == "False"
    assert normalize_pred("C") == "Uncertain"
    assert normalize_pred("A) True") == "True"
    assert normalize_pred("C) Uncertain") == "Uncertain"
    assert normalize_pred('{"reasoning": "...", "answer": "False"}') == "False"
    assert normalize_pred('```json\n{\n "reasoning": "x", "answer": "Uncertain"') == "Uncertain"
    assert normalize_pred("") == "PARSE_FAIL"
    assert normalize_pred("ERROR") == "PARSE_FAIL"
    assert normalize_pred("I am unable to answer this question.") == "PARSE_FAIL"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
