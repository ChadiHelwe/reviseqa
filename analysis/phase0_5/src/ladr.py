"""LADR (Prover9/Mace4) toolkit for Phase 0.5.

- fol_to_ladr: unicode FOL (dataset `*_fol` fields) -> LADR syntax.
  Validated against every stored `prover9_input` in verified-400 (see
  validate_converter()).
- run_prover9 / run_mace4: subprocess wrappers with budget accounting.

Deterministic; no randomness.
"""
import re
import subprocess

PROVER9 = "/opt/homebrew/bin/prover9"
MACE4 = "/opt/homebrew/bin/mace4"


# ── FOL -> LADR conversion ────────────────────────────────────────────────
def _convert_xor(s: str) -> str:
    """Rewrite every `A ⊕ B` as `-(A <-> B)` using bracket matching."""
    while "⊕" in s:
        i = s.find("⊕")
        # left operand
        depth = 0
        left = i - 1
        while left > 0:
            c = s[left]
            if c == ")":
                depth += 1
            elif c == "(":
                if depth == 0:
                    left += 1
                    break
                depth -= 1
            left -= 1
        # right operand
        depth = 0
        right = i + 1
        while right < len(s):
            c = s[right]
            if c == "(":
                depth += 1
            elif c == ")":
                if depth == 0:
                    break
                depth -= 1
            right += 1
        inner = s[left:right].replace("⊕", "<->", 1)
        s = f"{s[:left]}(-({inner.strip()})){s[right:]}"
    return s


def fol_to_ladr(s: str) -> str:
    s = _convert_xor(s)
    for a, b in [("¬", "-"), ("∧", "&"), ("∨", "|"), ("→", "->"),
                 ("↔", "<->"), ("⟷", "<->"), ("≠", "!=")]:
        s = s.replace(a, b)
    s = re.sub(r"∀(\w+)", r"all \1", s)
    s = re.sub(r"∃(\w+)", r"exists \1", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_formula(s: str) -> str:
    """Paren-insensitive normalization for set comparison (drop final period,
    whitespace, and all parentheses). Coarse by design: string style differs
    between our converter and the nltk-canonicalized stored inputs; logical
    equivalence of differing pairs is checked separately with prover9
    (validate_converter)."""
    s = s.strip()
    if s.endswith("."):
        s = s[:-1]
    return re.sub(r"[\s()]+", "", s)


def as_input_lines(formulas):
    out = []
    for f in formulas:
        f = f.strip()
        if not f.endswith("."):
            f = f + " ."
        out.append(f)
    return out


def build_input(assumptions, goal=None, max_seconds=None):
    parts = []
    if max_seconds is not None:
        parts.append(f"assign(max_seconds, {int(max_seconds)}).")
    parts.append("formulas(assumptions).")
    parts.extend(as_input_lines(assumptions))
    parts.append("end_of_list.")
    if goal is not None:
        parts.append("formulas(goals).")
        parts.extend(as_input_lines([goal]))
        parts.append("end_of_list.")
    return "\n".join(parts) + "\n"


# ── Runners ───────────────────────────────────────────────────────────────
def _run(cmd, inp, wall_timeout):
    """Binary-safe subprocess run (LADR may echo truncated UTF-8 bytes)."""
    r = subprocess.run(cmd, input=inp.encode("ascii"), capture_output=True,
                       timeout=wall_timeout)
    return r.returncode, r.stdout.decode("utf-8", errors="replace"), \
        r.stderr.decode("utf-8", errors="replace")


def run_prover9(assumptions, goal=None, max_seconds=600):
    """Returns (status, seconds_user). status:
    PROVED | EXHAUSTED (sos empty: no proof exists) | TIMEOUT | LIMIT | ERROR.
    With goal=None, PROVED means the assumptions are contradictory."""
    inp = build_input(assumptions, goal, max_seconds)
    try:
        rc, out, err = _run([PROVER9], inp, max_seconds + 60)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", float(max_seconds)
    except UnicodeEncodeError:
        return "ERROR", float("nan")
    m = re.search(r"User_CPU=(\d+\.\d+)", out)
    secs = float(m.group(1)) if m else float("nan")
    if "THEOREM PROVED" in out:
        return "PROVED", secs
    if rc == 2 or "sos_empty" in out:
        return "EXHAUSTED", secs
    if rc == 4 or "max_seconds" in out:
        return "TIMEOUT", secs
    if rc in (3, 5, 6):
        return "LIMIT", secs
    return "ERROR", secs


def run_mace4(assumptions, goal=None, max_seconds=60, max_domain=20):
    """Returns (status, seconds). status:
    MODEL (found; with goal: countermodel certifying non-entailment;
    without goal: consistency certificate) |
    EXHAUSTED (no model up to max_domain) | TIMEOUT | ERROR."""
    inp = build_input(assumptions, goal)
    try:
        rc, out, err = _run([MACE4, "-t", str(int(max_seconds)), "-N", str(max_domain)],
                            inp, max_seconds + 60)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", float(max_seconds)
    except UnicodeEncodeError:
        return "ERROR", float("nan")
    m = re.search(r"User_CPU=(\d+\.\d+)", out)
    secs = float(m.group(1)) if m else float("nan")
    if "interpretation(" in out or rc == 0:
        return "MODEL", secs
    if "all_domain_sizes" in out + err or rc == 2:
        return "EXHAUSTED", secs
    if rc in (4, 5) or "max_sec" in out + err:
        return "TIMEOUT", secs
    return "ERROR", secs


# ── Gold re-derivation ────────────────────────────────────────────────────
def rederive_gold(assumptions, goal_q, p9_budget=600, m4_budget=60):
    """Full status battery for one turn state. Returns dict."""
    out = {}
    out["consistency_mace4"], out["t_consistency"] = run_mace4(assumptions, None, m4_budget)
    out["contradiction_prover9"], out["t_contradiction"] = run_prover9(
        assumptions, None, p9_budget)
    out["q_prover9"], out["t_q"] = run_prover9(assumptions, goal_q, p9_budget)
    out["negq_prover9"], out["t_negq"] = run_prover9(
        assumptions, f"-({norm_strip_period(goal_q)})", p9_budget)
    # countermodel certificates of unprovability
    out["q_countermodel"], _ = run_mace4(assumptions, goal_q, m4_budget)
    out["negq_countermodel"], _ = run_mace4(
        assumptions, f"-({norm_strip_period(goal_q)})", m4_budget)

    inconsistent = out["contradiction_prover9"] == "PROVED"
    if inconsistent:
        gold = "INCONSISTENT"
    elif out["q_prover9"] == "PROVED":
        gold = "True"
    elif out["negq_prover9"] == "PROVED":
        gold = "False"
    elif (out["q_prover9"] in ("EXHAUSTED",) or out["q_countermodel"] == "MODEL") and \
         (out["negq_prover9"] in ("EXHAUSTED",) or out["negq_countermodel"] == "MODEL"):
        gold = "Uncertain"
    else:
        gold = "UNRESOLVED"
    out["rederived_gold"] = gold
    return out


def norm_strip_period(s: str) -> str:
    s = s.strip()
    return s[:-1].strip() if s.endswith(".") else s


def validate_converter(data_dir, equiv_sample=150, seed=464):
    """Two-layer validation of fol_to_ladr against every stored prover9_input.

    Layer 1 (all edits): paren-insensitive set equality of assumption lists.
    Layer 2 (sampled pairs whose exact strings differ): prover9 proof that
    `(mine) <-> (stored)` is a tautology.
    Returns (n_ok, n_bad, bad_list, n_equiv_ok, n_equiv_bad).
    """
    import json
    import os
    import random
    n_ok = n_bad = 0
    bad = []
    differing_pairs = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(data_dir, fname)) as f:
            d = json.load(f)
        for i, e in enumerate(d.get("edits", []), 1):
            mine_map = {norm_formula(fol_to_ladr(s)): fol_to_ladr(s)
                        for s in e["edited_context_fol"]}
            stored_map = {norm_formula(s): norm_strip_period(s)
                          for s in e["prover9_input"]["formulas(assumptions)"]}
            if set(mine_map) == set(stored_map):
                n_ok += 1
                for key in mine_map:
                    if re.sub(r"\s+", "", mine_map[key]) != re.sub(r"\s+", "", stored_map[key]):
                        differing_pairs.append((mine_map[key], stored_map[key]))
            else:
                n_bad += 1
                if len(bad) < 5:
                    bad.append((fname, i, sorted(set(mine_map) - set(stored_map))[:2],
                                sorted(set(stored_map) - set(mine_map))[:2]))
    rng = random.Random(seed)
    uniq = sorted(set(differing_pairs))
    sample = rng.sample(uniq, min(equiv_sample, len(uniq)))
    n_equiv_ok = n_equiv_bad = 0
    for mine, stored in sample:
        status, _ = run_prover9([], f"({mine}) <-> ({stored})", max_seconds=30)
        if status == "PROVED":
            n_equiv_ok += 1
        else:
            n_equiv_bad += 1
    return n_ok, n_bad, bad, n_equiv_ok, n_equiv_bad


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", "..", ".."))
    ok, bad, ex, eq_ok, eq_bad = validate_converter(
        os.path.join(repo, "reviseqa_data", "nl", "verified-400"))
    print(f"layer 1 (paren-insensitive set match): {ok} ok, {bad} bad")
    for b in ex:
        print(" ", b)
    print(f"layer 2 (prover9 equivalence on sampled differing pairs): "
          f"{eq_ok} proved, {eq_bad} failed")
