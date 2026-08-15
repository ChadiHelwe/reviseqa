"""Edit coding and answer normalization for ReviseQA Phase 0.

All rules here are deterministic; no randomness.
"""
import json
import re

CANONICAL = {"True", "False", "Uncertain"}
LETTER_MAP = {"A": "True", "B": "False", "C": "Uncertain"}
SHORT = {"True": "T", "False": "F", "Uncertain": "U"}

# "answer" field inside a (possibly truncated / malformed) JSON blob
_JSON_ANSWER_RE = re.compile(
    r'"answer"\s*:\s*"?\s*(True|False|Uncertain|A|B|C)\b', re.IGNORECASE
)
# "Answer: True" plain-text style
_TEXT_ANSWER_RE = re.compile(
    r"\banswer\b\s*[:=]?\s*\"?\s*(True|False|Uncertain)\b", re.IGNORECASE
)


def normalize_pred(raw) -> str:
    """Map a raw prediction string to {True, False, Uncertain, PARSE_FAIL}.

    Handles: canonical strings, lowercase variants, bare option letters
    (A/B/C), "A) True"-style options, and JSON blobs containing an
    "answer" field. API-error sentinels ("ERROR") and empty strings are
    PARSE_FAIL.
    """
    if raw is None:
        return "PARSE_FAIL"
    s = str(raw).strip()
    if not s or s == "ERROR":
        return "PARSE_FAIL"
    # canonical / case variants
    cap = s.capitalize()
    if cap in CANONICAL:
        return cap
    # bare letter or "A)" / "A) True" style
    m = re.fullmatch(r"([ABC])\)?(?:\s*\)?\s*(True|False|Uncertain))?\.?", s, re.IGNORECASE)
    if m:
        letter = LETTER_MAP[m.group(1).upper()]
        if m.group(2) and m.group(2).capitalize() != letter:
            return "PARSE_FAIL"  # inconsistent letter/word pair
        return letter
    # JSON blob (possibly truncated): try proper parse first, then regex
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict) and "answer" in parsed:
            return normalize_pred(parsed["answer"])
    except (json.JSONDecodeError, TypeError):
        pass
    m = _JSON_ANSWER_RE.search(s)
    if m:
        tok = m.group(1)
        return LETTER_MAP.get(tok.upper(), tok.capitalize())
    m = _TEXT_ANSWER_RE.search(s)
    if m:
        return m.group(1).capitalize()
    return "PARSE_FAIL"


def edit_counts(edits_made: dict):
    """(n_add, n_rem) from a dataset `edits_made` dict."""
    n_add = len(edits_made.get("added_facts") or []) + len(edits_made.get("added_rules") or [])
    n_rem = len(edits_made.get("removed_facts") or []) + len(edits_made.get("removed_rules") or [])
    return n_add, n_rem


def classify_edit(edits_made: dict) -> str:
    """`add_only` (n_rem=0, n_add>0), `removal` (n_rem>0), `none`."""
    n_add, n_rem = edit_counts(edits_made)
    if n_rem > 0:
        return "removal"
    if n_add > 0:
        return "add_only"
    return "none"


def edit_target(edits_made: dict) -> str:
    """fact / rule / both / none, over the union of added+removed."""
    facts = bool(edits_made.get("added_facts")) or bool(edits_made.get("removed_facts"))
    rules = bool(edits_made.get("added_rules")) or bool(edits_made.get("removed_rules"))
    if facts and rules:
        return "both"
    if facts:
        return "fact"
    if rules:
        return "rule"
    return "none"


def transition(gold_prev: str, gold_t: str) -> str:
    """e.g. 'T→F', 'U→T', 'T→T'."""
    return f"{SHORT.get(gold_prev, '?')}→{SHORT.get(gold_t, '?')}"


_CONNECTIVE_RE = re.compile(r"[→↔⊕∨∧∀∃]")


def fol_kind(statement: str) -> str:
    """Syntactic fact/rule classification of a FOL statement.

    Any connective or quantifier ⇒ rule; bare (possibly negated) literal ⇒
    fact. More reliable than the dataset's own lists (≈21% of recorded
    "facts" contain connectives).
    """
    return "rule" if _CONNECTIVE_RE.search(statement) else "fact"


def classify_actual(added: set, removed: set):
    """Edit coding from the *actual* FOL context delta (cur − prev / prev − cur).

    Returns dict with n_add, n_rem, edit_class, edit_target.
    """
    n_add, n_rem = len(added), len(removed)
    if n_rem > 0:
        cls = "removal"
    elif n_add > 0:
        cls = "add_only"
    else:
        cls = "none"
    kinds = {fol_kind(s) for s in added} | {fol_kind(s) for s in removed}
    if kinds == {"fact"}:
        tgt = "fact"
    elif kinds == {"rule"}:
        tgt = "rule"
    elif kinds:
        tgt = "both"
    else:
        tgt = "none"
    return {"n_add": n_add, "n_rem": n_rem, "edit_class": cls, "edit_target": tgt}
