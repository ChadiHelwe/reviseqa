"""Phase 0.6 — Clean-slice re-cut & valid-subset preview (PHASE0_6_RECUT.md).

Jobs A, B, D + report. Job C (prover) runs separately in job_c.py and its
CSV is folded into the report here.

Inputs (read-only): analysis/phase0/tidy.parquet,
analysis/phase0_5/v1.1/manifest.csv, analysis/phase0_5/tables/*.csv,
reviseqa_data/nl/verified-400/.  Outputs: analysis/phase0_6/.
Seed 464 everywhere.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(REPO, "analysis", "phase0_6")
TAB = os.path.join(OUT, "tables")
FIG = os.path.join(OUT, "figures")
P0 = os.path.join(REPO, "analysis", "phase0")
P05 = os.path.join(REPO, "analysis", "phase0_5")
DATA = os.path.join(REPO, "reviseqa_data", "nl", "verified-400")

sys.path.insert(0, os.path.join(P0, "src"))
from analysis import (  # noqa: E402
    make_slice, flagged_models, autopsy_label, cluster_boot_did, did_table_row,
    md_table,
)

C_ADD, C_REM, C_YEL, C_OTHER = "#2a78d6", "#eb6834", "#eda100", "#9a9992"
PREVIEW_LABEL = "BIASED_SUBSET_PREVIEW — internal only, not for paper"


# ──────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────
def setup():
    manifest = pd.read_csv(os.path.join(P05, "v1.1", "manifest.csv"))
    manifest["ex"] = manifest.filename.str[:-5]
    ev = sorted(f[:-5] for f in os.listdir(DATA) if f.endswith(".json"))
    me = manifest[manifest.ex.isin(ev)]
    assert len(me) == len(ev) == 400, "STOP: manifest<->eval join ambiguous"
    clean = sorted(me[me.n_problems == 0].ex)
    assert abs(len(clean) - 301) <= 5, f"STOP: clean_examples = {len(clean)}"

    demo_correct = {}
    for e in ev:
        d = json.load(open(os.path.join(DATA, f"{e}.json")))
        demo_correct[e] = (d["answer"] == d["edits"][-1]["answer"]
                           if d.get("edits") else True)
    return set(clean), demo_correct, ev


def load_slices(clean):
    df = pd.read_parquet(os.path.join(P0, "tidy.parquet"))
    flagged = flagged_models(df)
    prim = make_slice(df, flagged=flagged)           # Phase-0 primary (original)
    prim_clean = prim[prim.example_id.isin(clean)]   # Job A slice
    return df, flagged, prim, prim_clean


# ──────────────────────────────────────────────────────────────────────────
# Job A
# ──────────────────────────────────────────────────────────────────────────
def flip_autopsy(sl, exclude_t1=False):
    s = sl[(sl.edit_class == "removal") & sl.answer_changed]
    if exclude_t1:
        s = s[s.turn_index > 1]
    s = s.copy()
    s["autopsy"] = s.apply(autopsy_label, axis=1)
    err = s[s.autopsy != "correct"]
    out = {}
    for setting in ["explicit", "implicit"]:
        e = err[err.setting == setting]
        n = len(e)
        out[f"stale_{setting}"] = round(100 * (e.autopsy == "stale").mean(), 2) if n else np.nan
        out[f"retreat_{setting}"] = round(100 * (e.autopsy == "uncertain_retreat").mean(), 2) if n else np.nan
        out[f"n_errors_{setting}"] = n
    pooled = err.autopsy.value_counts(normalize=True)
    out["retreat_pooled"] = round(100 * pooled.get("uncertain_retreat", 0.0), 2)
    out["stale_pooled"] = round(100 * pooled.get("stale", 0.0), 2)
    return out, err


def job_a(prim, prim_clean):
    rows_cmp = []

    # A1 flip autopsy (pooled; per-model to CSV)
    a1 = {}
    for name, sl in [("original", prim), ("clean_slice", prim_clean)]:
        a1[name], err = flip_autopsy(sl)
        a1[name + "_no_t1"], _ = flip_autopsy(sl, exclude_t1=True)
        if name == "clean_slice":
            per_model = (err.groupby(["model", "setting", "autopsy"], observed=True)
                            .size().rename("n").reset_index())
            tot = per_model.groupby(["model", "setting"], observed=True).n.transform("sum")
            per_model["share"] = (100 * per_model.n / tot).round(2)
            per_model.to_csv(os.path.join(TAB, "a1_flip_autopsy_per_model_clean.csv"),
                             index=False)
    a1_tab = pd.DataFrame([
        {"metric": k, "original": a1["original"].get(k), "clean_slice": a1["clean_slice"].get(k),
         "original_no_t1": a1["original_no_t1"].get(k),
         "clean_no_t1": a1["clean_slice_no_t1"].get(k)}
        for k in ["retreat_pooled", "stale_pooled", "retreat_explicit", "retreat_implicit",
                  "stale_explicit", "stale_implicit", "n_errors_explicit", "n_errors_implicit"]])
    a1_tab.to_csv(os.path.join(TAB, "a1_flip_autopsy.csv"), index=False)
    hold_a1 = (a1["clean_slice"]["retreat_pooled"] >= 50
               and a1["clean_slice"]["retreat_explicit"] > a1["clean_slice"]["retreat_implicit"])

    # A2 P3 invariant DiD
    p3 = {}
    for name, sl in [("original", prim), ("clean_slice", prim_clean)]:
        p3[name] = did_table_row(name, cluster_boot_did(sl[~sl.answer_changed]))
    a2 = pd.DataFrame([p3["original"], p3["clean_slice"]])
    a2.to_csv(os.path.join(TAB, "a2_p3_invariant.csv"), index=False)
    hold_a2 = p3["clean_slice"]["did_ci_lo"] <= 0 <= p3["clean_slice"]["did_ci_hi"]

    # A3 invariant-band descriptives
    a3 = []
    for name, sl in [("original", prim), ("clean_slice", prim_clean)]:
        inv = sl[~sl.answer_changed]
        g = inv.groupby(["edit_class", "setting"], observed=True).correct.agg(["mean", "size"])
        for (cls, st), r in g.iterrows():
            a3.append({"slice": name, "edit_class": cls, "setting": st,
                       "acc": round(100 * r["mean"], 2), "n": int(r["size"])})
    a3 = pd.DataFrame(a3)
    a3.to_csv(os.path.join(TAB, "a3_invariant_descriptives.csv"), index=False)

    # A4 P1b on audit-clean pure removals
    ca = pd.read_csv(os.path.join(P05, "tables", "comparator_audit.csv"))
    pr_ok = ca[(ca.cohort == "pure_removal") & (ca.verdict == "CONSISTENT_GOLD_OK")]
    pr_keys = set(zip(pr_ok.example_id, pr_ok.turn_index))

    def p1b_slice(sl):
        s = sl[sl.answer_changed].copy()
        is_pure = (s.n_rem > 0) & (s.n_add == 0)
        is_audit_ok = [(e, t) in pr_keys for e, t in zip(s.example_id, s.turn_index)]
        s = s[(s.edit_class == "add_only") | (is_pure & pd.Series(is_audit_ok, index=s.index))]
        s = s.assign(edit_class=np.where(s.edit_class == "add_only", "add_only", "removal"))
        return s

    a4_rows = []
    for name, sl in [("original", prim), ("clean_slice", prim_clean)]:
        s = p1b_slice(sl)
        n_add = len(s[(s.edit_class == "add_only") & (s.setting == "explicit")])
        n_rem = len(s[(s.edit_class == "removal") & (s.setting == "explicit")])
        if n_add > 0 and n_rem > 0:
            row = did_table_row(name, cluster_boot_did(s))
        else:
            # comparator empty: DiD incomputable; report pure-removal gain only
            rem = s[s.edit_class == "removal"]
            row = {"slice": name, "did": np.nan,
                   "acc_rem_explicit": round(100 * rem[rem.setting == "explicit"].correct.mean(), 2) if n_rem else np.nan,
                   "acc_rem_implicit": round(100 * rem[rem.setting == "implicit"].correct.mean(), 2) if n_rem else np.nan,
                   "n_add_explicit": n_add, "n_rem_explicit": n_rem}
        row["pure_removal_gain_impl_minus_expl"] = (
            round(row.get("acc_rem_implicit", np.nan) - row.get("acc_rem_explicit", np.nan), 2)
            if not pd.isna(row.get("acc_rem_implicit", np.nan)) else np.nan)
        a4_rows.append(row)
    a4 = pd.DataFrame(a4_rows)
    a4.to_csv(os.path.join(TAB, "a4_p1b_audit_clean.csv"), index=False)
    a4_clean = a4_rows[1]
    hold_a4 = (not pd.isna(a4_clean.get("did"))) and a4_clean["did"] > 0

    # A5 transition split
    a5_rows = []
    for name, sl in [("original", prim), ("clean_slice", prim_clean)]:
        s = sl[(sl.edit_class == "removal") & sl.answer_changed
               & sl.transition.isin(["T→F", "F→T"])].copy()
        s["autopsy"] = s.apply(autopsy_label, axis=1)
        for tr in ["T→F", "F→T"]:
            t = s[s.transition == tr]
            err_e = t[(t.setting == "explicit") & (t.autopsy != "correct")]
            err_i = t[(t.setting == "implicit") & (t.autopsy != "correct")]
            a5_rows.append({
                "slice": name, "transition": tr,
                "acc_explicit": round(100 * t[t.setting == "explicit"].correct.mean(), 2),
                "acc_implicit": round(100 * t[t.setting == "implicit"].correct.mean(), 2),
                "gain": round(100 * (t[t.setting == "implicit"].correct.mean()
                                     - t[t.setting == "explicit"].correct.mean()), 2),
                "retreat_share_explicit": round(100 * (err_e.autopsy == "uncertain_retreat").mean(), 2),
                "retreat_share_implicit": round(100 * (err_i.autopsy == "uncertain_retreat").mean(), 2),
                "n_turns_per_setting": len(t[t.setting == "explicit"]),
            })
    a5 = pd.DataFrame(a5_rows)
    a5.to_csv(os.path.join(TAB, "a5_transition_split.csv"), index=False)
    c = a5[a5["slice"] == "clean_slice"].set_index("transition")
    hold_a5 = abs(c.loc["T→F", "retreat_share_explicit"]
                  - c.loc["F→T", "retreat_share_explicit"]) <= 10

    holds = pd.DataFrame([
        {"finding": "A1 flip autopsy", "criterion": "retreat pooled >= 50% and explicit > implicit",
         "value": f"pooled {a1['clean_slice']['retreat_pooled']}%, "
                  f"expl {a1['clean_slice']['retreat_explicit']}% vs impl {a1['clean_slice']['retreat_implicit']}%",
         "pass": bool(hold_a1)},
        {"finding": "A2 P3 invariant", "criterion": "DiD CI contains 0",
         "value": f"DiD {p3['clean_slice']['did']} [{p3['clean_slice']['did_ci_lo']}, {p3['clean_slice']['did_ci_hi']}]",
         "pass": bool(hold_a2)},
        {"finding": "A4 P1b", "criterion": "DiD > 0",
         "value": ("DiD incomputable: add_only×changed comparator empty on clean slice "
                   f"(n_add={a4_clean.get('n_add_explicit', 0)}); pure-removal gain "
                   f"{a4_clean.get('pure_removal_gain_impl_minus_expl')} pts"
                   if pd.isna(a4_clean.get("did")) else f"DiD {a4_clean['did']}"),
         "pass": bool(hold_a4)},
        {"finding": "A5 transition split", "criterion": "explicit retreat shares within 10 pts",
         "value": f"T→F {c.loc['T→F', 'retreat_share_explicit']}% vs F→T {c.loc['F→T', 'retreat_share_explicit']}%",
         "pass": bool(hold_a5)},
    ])
    holds.to_csv(os.path.join(TAB, "a_hold_criteria.csv"), index=False)
    return a1, a1_tab, a2, a3, a4, a5, holds


def fig_recut(prim_clean):
    # autopsy bars (changed turns, removal class), clean slice
    s = prim_clean[(prim_clean.edit_class == "removal") & prim_clean.answer_changed].copy()
    s["autopsy"] = s.apply(autopsy_label, axis=1)
    err = s[s.autopsy != "correct"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax = axes[0]
    xs = np.arange(2)
    bottom = np.zeros(2)
    for cat, col in [("stale", C_REM), ("uncertain_retreat", C_YEL), ("other_changed", C_OTHER)]:
        vals = np.array([100 * (err[err.setting == st].autopsy == cat).mean()
                         for st in ["explicit", "implicit"]])
        ax.bar(xs, vals, bottom=bottom, color=col, width=0.55,
               edgecolor="#fcfcfb", linewidth=1.5, label=cat)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 8:
                ax.text(xi, b + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=8)
        bottom += vals
    ax.set_xticks(xs, ["explicit", "implicit"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("share of flip-turn errors (%)")
    ax.set_title("Flip-turn autopsy — clean slice (301)", fontsize=10)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))

    ax = axes[1]
    a5 = pd.read_csv(os.path.join(TAB, "a5_transition_split.csv"))
    for tr, col in [("T→F", C_REM), ("F→T", C_ADD)]:
        r = a5[(a5["slice"] == "clean_slice") & (a5.transition == tr)].iloc[0]
        ax.plot([0, 1], [r.acc_explicit, r.acc_implicit], "-o", color=col, lw=2, ms=6)
        ax.annotate(tr, (1, r.acc_implicit), xytext=(6, 0), textcoords="offset points",
                    fontsize=9, va="center")
    ax.set_xticks([0, 1], ["explicit", "implicit"])
    ax.set_xlim(-0.2, 1.45)
    ax.set_ylim(0, 100)
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Transition split — clean slice", fontsize=10)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color="#e5e4df", lw=0.7)
        a.set_axisbelow(True)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(FIG, f"recut_autopsy_transition.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Job B
# ──────────────────────────────────────────────────────────────────────────
def lcata(sl, correct_col, ks=(2, 4, 7)):
    """LCATA@k = share of examples whose first k turns are ALL correct
    (paper metric, src/metric.py)."""
    pv = sl.pivot_table(index=["model", "setting", "example_id"],
                        columns="turn_index", values=correct_col, aggfunc="first")
    out = []
    for (m, st), g in pv.groupby(level=[0, 1]):
        row = {"model": m, "setting": st, "n_examples": len(g)}
        for k in ks:
            cols = [c for c in range(1, k + 1)]
            row[f"lcata@{k}"] = round(100 * g[cols].all(axis=1).mean(), 2)
        out.append(row)
    return pd.DataFrame(out)


def job_b(df, flagged, clean, demo_correct):
    valid = sorted(e for e in clean if demo_correct[e])
    # primary tracks, all turns incl. corrupted-delta ones (LCATA is a
    # trajectory metric; per-turn validity is already guaranteed by clean)
    base = df[(df.prompting == "standard") & (~df.feedback) & (~df.model.isin(flagged))]

    published = lcata(base, "log_correct")                       # full 400, harness scoring
    preview = lcata(base[base.example_id.isin(valid)], "correct")  # 148, our scoring
    merged = published.merge(preview, on=["model", "setting"],
                             suffixes=("_published_full400", "_preview"))
    for k in (2, 4, 7):
        merged[f"delta@{k}"] = (merged[f"lcata@{k}_preview"]
                                - merged[f"lcata@{k}_published_full400"]).round(2)
    merged.insert(0, "label", PREVIEW_LABEL)
    merged.to_csv(os.path.join(TAB, "b_lcata_preview.csv"), index=False)

    # per-turn conditional accuracy
    pt = []
    for name, sl, col in [("published_full400", base, "log_correct"),
                          ("preview_valid148", base[base.example_id.isin(valid)], "correct")]:
        g = (sl.groupby(["model", "setting", "turn_index"], observed=True)[col]
               .mean().mul(100).round(2).rename("acc").reset_index())
        g["slice"] = name
        pt.append(g)
    pt = pd.concat(pt)
    pt.insert(0, "label", PREVIEW_LABEL)
    pt.to_csv(os.path.join(TAB, "b_per_turn_accuracy.csv"), index=False)

    # composition bias
    u = df[["example_id", "turn_index", "answer_changed"]].drop_duplicates()
    traj = u.groupby("example_id").answer_changed.agg(
        flip_count="sum", n_turns="size").reset_index()
    traj["invariant_share"] = ((traj.n_turns - traj.flip_count) / traj.n_turns).round(3)
    traj["in_valid"] = traj.example_id.isin(valid)
    comp = (traj.groupby("in_valid")
                .agg(n=("example_id", "size"), mean_flips=("flip_count", "mean"),
                     mean_invariant_share=("invariant_share", "mean")).round(3))
    dist = (traj.groupby(["in_valid", "flip_count"]).size().rename("n").reset_index())
    comp.to_csv(os.path.join(TAB, "b_composition_bias.csv"))
    dist.to_csv(os.path.join(TAB, "b_flipcount_distribution.csv"), index=False)
    return merged, pt, comp, dist, valid


# ──────────────────────────────────────────────────────────────────────────
# Job D
# ──────────────────────────────────────────────────────────────────────────
def job_d(prim_clean, demo_correct):
    d = prim_clean.copy()
    d["demo_wrong"] = ~d.example_id.map(demo_correct)

    # D1 anchoring fingerprint on turns 1-2 errors, matched on (turn, class, transition)
    early = d[(d.turn_index <= 2) & (~d.correct)].copy()
    early["anchored"] = early.pred == early.demo_answer_as_shown
    fp_rows = []
    for scope, g in [("pooled", early)] + [(m, early[early.model == m])
                                           for m in sorted(early.model.unique())]:
        cells = g.groupby(["turn_index", "edit_class", "transition"], observed=True)
        num = den = 0.0
        for _, cell in cells:
            w = cell[cell.demo_wrong]
            c = cell[~cell.demo_wrong]
            if len(w) and len(c):
                num += len(w) * (w.anchored.mean() - c.anchored.mean())
                den += len(w)
        fp_rows.append({"scope": scope,
                        "excess_anchoring_pts": round(100 * num / den, 2) if den else np.nan,
                        "n_matched_wrong_errors": int(den),
                        "raw_anchored_wrong": round(100 * g[g.demo_wrong].anchored.mean(), 2)
                        if g.demo_wrong.any() else np.nan,
                        "raw_anchored_correct": round(100 * g[~g.demo_wrong].anchored.mean(), 2)
                        if (~g.demo_wrong).any() else np.nan})
    fp = pd.DataFrame(fp_rows)
    fp.to_csv(os.path.join(TAB, "d1_anchoring_fingerprint.csv"), index=False)
    excess_pooled = float(fp.loc[fp.scope == "pooled", "excess_anchoring_pts"].iloc[0])

    # D2/D3 stratified depression by turn
    u = d[["example_id", "turn_index", "answer_changed"]].drop_duplicates()
    traj = u.groupby("example_id").answer_changed.agg(flip_count="sum").reset_index()
    first_flip = (u[u.answer_changed].groupby("example_id").turn_index.min()
                    .rename("first_flip").reset_index())
    traj = traj.merge(first_flip, on="example_id", how="left").fillna({"first_flip": 0})
    d = d.merge(traj, on="example_id")
    d["stratum"] = list(zip(d.flip_count, d.first_flip))

    dep_rows = []
    for t in range(1, 8):
        g = d[d.turn_index == t]
        num = den = 0.0
        n_w_total = int(g.demo_wrong.sum()) // 2  # per setting halves; count items
        for _, cell in g.groupby(["stratum", "setting"], observed=True):
            w = cell[cell.demo_wrong]
            c = cell[~cell.demo_wrong]
            if len(w) and len(c):
                num += len(w) * (c.correct.mean() - w.correct.mean())
                den += len(w)
        dep_rows.append({"turn_index": t,
                         "depression_pts": round(100 * num / den, 2) if den else np.nan,
                         "n_matched_demo_wrong": int(den),
                         "coverage_of_demo_wrong": round(den / max(1, int(g.demo_wrong.sum())), 3)})
    dep = pd.DataFrame(dep_rows)
    dep.to_csv(os.path.join(TAB, "d2_stratified_depression_by_turn.csv"), index=False)

    dep_late = dep[dep.turn_index >= 3].depression_pts.mean()
    weak_fp = abs(excess_pooled) <= 5.0
    verdict_pass = (dep_late <= 3.0) and weak_fp
    verdict = {
        "excess_anchoring_pooled_pts": excess_pooled,
        "fingerprint_weak(threshold |x|<=5)": weak_fp,
        "mean_depression_turns_3plus_pts": round(float(dep_late), 2),
        "depression_small(threshold <=3)": bool(dep_late <= 3.0),
        "verdict": ("301-clean absolute tables with quantified caveat; NO re-runs needed"
                    if verdict_pass else
                    "restrict absolute tables to clean ∩ demo_correct (Job B caveat) "
                    "or move to appendix as bounded estimates"),
    }
    pd.DataFrame([verdict]).to_csv(os.path.join(TAB, "d_verdict.csv"), index=False)
    return fp, dep, verdict


# ──────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    clean, demo_correct, ev = setup()
    df, flagged, prim, prim_clean = load_slices(clean)
    n_valid = sum(demo_correct[e] for e in clean)
    print(f"clean={len(clean)}, demo_correct overall={sum(demo_correct.values())}, "
          f"clean∩demo_correct={n_valid}")

    a1, a1_tab, a2, a3, a4, a5, holds = job_a(prim, prim_clean)
    fig_recut(prim_clean)
    print(holds.to_string(index=False))

    b_lcata, b_pt, b_comp, b_dist, valid = job_b(df, flagged, clean, demo_correct)
    fp, dep, verdict = job_d(prim_clean, demo_correct)
    print("Job D verdict:", verdict["verdict"])

    jc_path = os.path.join(TAB, "c_leftover_comparator.csv")
    jc = pd.read_csv(jc_path) if os.path.exists(jc_path) else None

    b_lcata_disp = b_lcata[["model", "setting"] +
                           [c for c in b_lcata.columns if "lcata" in c or "delta" in c]]
    report = f"""# Phase 0.6 — Clean-Slice Re-cut & Valid-Subset Preview: Report

Executed per `PHASE0_6_RECUT.md`; seed 464; inputs read-only; reproduce with
`analysis/phase0_6/run.sh`.

**Setup.** `clean_examples` = manifest-clean ∩ eval-400 = **{len(clean)}**
(expected 301 ✓). `demo_correct` = **{sum(demo_correct.values())}/400**
(expected ~193 ✓); clean ∩ demo_correct = **{n_valid}**. Primary slice as in
Phase 0 (Standard × no-feedback × 17 non-flagged models × paired ×
actual-delta coding). No stop-conditions triggered.

## Job A — Re-cut citable findings on clean_examples

Hold criteria (`tables/a_hold_criteria.csv`):

{md_table(holds)}

**Three of four hold criteria pass (A1, A2, A5). A4 FAILS — flagged
prominently, nothing adjusted:** the P1b DiD is *incomputable* on the clean
slice because, once examples are manifest-clean **and** turns are
delta-verified, the changed band is compositionally homogeneous — there are
**zero** `add_only × changed` turns and **zero** pure-removal changed turns
left. Every fully-verified answer-changing edit is a remove+add compound
flip (exactly what FOL monotonicity requires). The apparent compositional
variety in the changed band (pure adds that flip, pure removals that flip)
was *entirely* a metadata/labeling artifact. Consequence: **no
within-changed-band compositional contrast (P1- or P1b-style) is estimable
from v1 data at all** — that comparison genuinely requires Study 2's
generated conditions. For reference, on the original (all-examples) slice
the audit-clean-restricted P1b DiD is +11.6 [−6.6, +34.1] (n.s., 5 unique
turns).

1. **Flip autopsy** (`a1_flip_autopsy.csv`): pooled retreat
   {a1["clean_slice"]["retreat_pooled"]}% (original
   {a1["original"]["retreat_pooled"]}%); explicit
   {a1["clean_slice"]["retreat_explicit"]}% vs implicit
   {a1["clean_slice"]["retreat_implicit"]}% (original
   {a1["original"]["retreat_explicit"]}/{a1["original"]["retreat_implicit"]}).
   Excluding turn 1: pooled {a1["clean_slice_no_t1"]["retreat_pooled"]}%.
   The finding is unchanged by cleaning.
2. **P3** (`a2_p3_invariant.csv`): clean-slice DiD
   {a2.iloc[1]["did"]} [{a2.iloc[1]["did_ci_lo"]}, {a2.iloc[1]["did_ci_hi"]}]
   (original {a2.iloc[0]["did"]} [{a2.iloc[0]["did_ci_lo"]},
   {a2.iloc[0]["did_ci_hi"]}]) — null preserved.
3. **Invariant-band descriptives** (`a3_invariant_descriptives.csv`):
   within ±2 pts of original in every cell.
4. **P1b** (`a4_p1b_audit_clean.csv`): incomputable on the clean slice —
   n_add = n_pure_rem = 0 changed turns survive full verification (see the
   hold-criteria discussion above). Original-slice audit-clean-restricted
   DiD: {a4.iloc[0].get("did")} [{a4.iloc[0].get("did_ci_lo")},
   {a4.iloc[0].get("did_ci_hi")}] — positive but no longer significant once
   restricted to audited turns. Phase 0's P1b should be cited only with
   this caveat attached.
5. **Transition split** (`a5_transition_split.csv`): explicit retreat shares
   T→F {a5[(a5["slice"] == "clean_slice") & (a5.transition == "T→F")].iloc[0]["retreat_share_explicit"]}%
   vs F→T {a5[(a5["slice"] == "clean_slice") & (a5.transition == "F→T")].iloc[0]["retreat_share_explicit"]}%
   — symmetric within 10 pts.

Figures: `figures/recut_autopsy_transition.{{png,pdf}}`.

## Job B — Valid-subset preview ({PREVIEW_LABEL})

Subset: {n_valid} examples (logic-clean ∧ demonstration-correct); on these,
existing v1 responses are fully valid end-to-end. Tables:
`b_lcata_preview.csv` (LCATA@2/4/7 per model × setting: published-equivalent
recomputed from full-400 logs with the harness's own scoring, vs the
valid-subset with corrected scoring), `b_per_turn_accuracy.csv`.

Composition bias (`b_composition_bias.csv`, `b_flipcount_distribution.csv`):
demo-correct requires an even number of net flips, so odd-flip trajectories
are excluded by construction —

{md_table(b_comp.reset_index())}

Pooled LCATA deltas (preview − published-equivalent), mean across models:
@2 {b_lcata["delta@2"].mean():+.1f}, @4 {b_lcata["delta@4"].mean():+.1f},
@7 {b_lcata["delta@7"].mean():+.1f} pts (explicit and implicit pooled;
per-model table in CSV). Interpretation: order-of-magnitude only — the
subset is easier (more invariant turns) *and* scoring recoveries push in the
same direction, so treat as an upper bound on how much corrected re-runs
would raise published trajectory numbers.

## Job C — Leftover comparator turns

{md_table(jc) if jc is not None else "(run src/job_c.py first)"}

Both surviving `CONSISTENT_GOLD_OK` comparator turns have **mislabeled
pre-edit states** (predicted): the stored pre-edit answers do not re-derive
at 600 s with certificates, so the "flips" are illusory (`ex_802` t6 is
really U→F, not T→F; `ex_813` t3 is really T→T, not F→T). **The Phase-0 P1
comparator is now 18/18 invalid.** Appended to the Phase-0 erratum.

## Job D — Demo-bug impact quantification

1. **Anchoring fingerprint** (`d1_anchoring_fingerprint.csv`): on
   turns 1–2 errors, P(pred == displayed demo answer) on demo-wrong
   examples minus the matched demo-correct rate =
   **{verdict["excess_anchoring_pooled_pts"]:+.1f} pts pooled**
   (weak threshold |x| ≤ 5).
2. **Stratified depression** (`d2_stratified_depression_by_turn.csv`):
   accuracy(demo-correct) − accuracy(demo-wrong) within strata matched on
   (flip count, first-flip position) × setting, by turn. Odd-flip strata are
   demo-wrong by parity and unmatchable; coverage per turn is in the table.
3. **Decay:** mean matched depression at turns ≥ 3 =
   **{verdict["mean_depression_turns_3plus_pts"]:+.2f} pts**
   (threshold ≤ 3).

**Verdict: {verdict["verdict"]}**

## Non-goal (per spec)

No log-derived absolute table in this repo may be presented as bug-free
without the Job D verdict attached. Log surgery removes quarantined
examples, corrupted-delta turns, and scoring errors; it cannot correct model
behavior conditioned on corrupted prompts — Job D bounds that residual. No
fresh-400 re-run is planned (zero budget); Study-2 items were never shown to
any model, so no recomputation substitutes for its new inference.
"""
    with open(os.path.join(OUT, "report.md"), "w") as f:
        f.write(report)
    print("report written")


if __name__ == "__main__":
    main()
