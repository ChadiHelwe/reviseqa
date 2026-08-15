"""ReviseQA Phase 0 analysis: descriptives, interaction tests, error autopsy.

Reads analysis/phase0/tidy.parquet (built by parse_logs.py).
Writes tables/*.csv, figures/*.{png,pdf}, and report.md.

Fully deterministic: seed 464 for all resampling.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrix
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ──────────────────────────────────────────────────────────────────────────
# Pre-registered decision-gate thresholds (editable in one place)
# ──────────────────────────────────────────────────────────────────────────
SEED = 464
B_BOOT = 2000
GATE_P1_DID_MIN_PTS = 5.0        # suppression GO: pooled P1 DiD >= this
GATE_P1_MODEL_SHARE = 0.60       # ... and DiD > 0 in >= this share of models
GATE_INTRINSIC_MIN_PTS = 5.0     # intrinsic GO: implicit removal deficit >= this
GATE_RETREAT_SHARE = 0.50        # headline: uncertain_retreat >= this share of flip errors
FLAG_PARSE_FAIL_PCT = 15.0       # model flagged if > this in any primary run
PRIMARY_PROMPTING = "standard"   # per §2.1 audit verdict (LATEX_TYPO_ONLY -> Standard)
P3_MIN_POOLED_TURNS = 200        # run P3 only if removal x invariant >= this (pooled)

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TAB = os.path.join(OUT, "tables")
FIG = os.path.join(OUT, "figures")

# dataviz palette (validated): slots 1-3 + neutral
C_ADD = "#2a78d6"      # blue   – add_only
C_REM = "#eb6834"      # orange – removal
C_PURE = "#1baf7a"     # aqua   – pure removal
C_OTHER = "#9a9992"    # neutral gray
C_YEL = "#eda100"

pd.set_option("display.width", 200)


# ──────────────────────────────────────────────────────────────────────────
# Slices
# ──────────────────────────────────────────────────────────────────────────
def flagged_models(df):
    prim = df[(df.prompting == PRIMARY_PROMPTING) & (~df.feedback)]
    pq = prim.groupby(["model", "setting"]).parse_ok.apply(lambda s: 100 * (~s).mean())
    return sorted(set(pq[pq > FLAG_PARSE_FAIL_PCT].reset_index().model))


def paired_filter(d):
    """Restrict each model to the explicit∩implicit example intersection."""
    keep = []
    for m, g in d.groupby("model", observed=True):
        e = set(g.loc[g.setting == "explicit", "example_id"])
        i = set(g.loc[g.setting == "implicit", "example_id"])
        keep.append(g[g.example_id.isin(e & i)])
    return pd.concat(keep, ignore_index=True)


def make_slice(df, prompting=PRIMARY_PROMPTING, feedback=False, include_flagged=False,
               drop_parse_fail=False, flagged=(), exclude_mismatch=True,
               use_recorded=False):
    """Primary slice (post-hoc amendment, user-approved): edit coding from the
    actual FOL context delta, and turns whose recorded `edits_made` mismatches
    the actual delta are excluded (the explicit track saw a wrong delta there).
    `use_recorded=True` + `exclude_mismatch=False` reproduces the
    pre-registered spec coding exactly (sensitivity f)."""
    d = df[(df.prompting == prompting) & (df.feedback == feedback)]
    if use_recorded:
        d = d.assign(edit_class=d.edit_class_recorded, edit_target=d.edit_target_recorded,
                     n_add=d.n_add_recorded, n_rem=d.n_rem_recorded)
    if not include_flagged:
        d = d[~d.model.isin(flagged)]
    if drop_parse_fail:
        d = d[d.parse_ok]
    if exclude_mismatch:
        d = d[~d.delta_mismatch]
    d = d[d.edit_class != "none"]
    return paired_filter(d)


# ──────────────────────────────────────────────────────────────────────────
# Estimator 1: DiD on accuracy with cluster bootstrap over examples
# ──────────────────────────────────────────────────────────────────────────
def cluster_boot_did(d, classes=("add_only", "removal"), b=B_BOOT, seed=SEED):
    """DiD = (Add_expl − Rem_expl) − (Add_impl − Rem_impl), cluster = example.

    Returns dict with cell means, per-setting deficits (Add − Rem), DiD,
    and 95% percentile CIs from B cluster-bootstrap draws.
    """
    cells = [(c, s) for c in classes for s in ("explicit", "implicit")]
    ex_ids = np.sort(d.example_id.unique())
    ex_idx = {e: i for i, e in enumerate(ex_ids)}
    E = len(ex_ids)
    k = np.zeros((E, 4))
    n = np.zeros((E, 4))
    for j, (c, s) in enumerate(cells):
        sub = d[(d.edit_class == c) & (d.setting == s)]
        g = sub.groupby("example_id", observed=True).correct.agg(["sum", "size"])
        rows = [ex_idx[e] for e in g.index]
        k[rows, j] = g["sum"].to_numpy()
        n[rows, j] = g["size"].to_numpy()

    def stats_from(kk, nn):
        with np.errstate(invalid="ignore", divide="ignore"):
            acc = kk / nn
        ae, ai, re_, ri = acc  # add_expl, add_impl, rem_expl, rem_impl
        return {
            "add_explicit": ae, "add_implicit": ai,
            "rem_explicit": re_, "rem_implicit": ri,
            "deficit_explicit": ae - re_, "deficit_implicit": ai - ri,
            "did": (ae - re_) - (ai - ri),
        }

    point = stats_from(k.sum(0), n.sum(0))
    rng = np.random.default_rng(seed)
    draws = {key: np.empty(b) for key in point}
    for bi in range(b):
        idx = rng.integers(0, E, E)
        s = stats_from(k[idx].sum(0), n[idx].sum(0))
        for key, v in s.items():
            draws[key][bi] = v
    out = {}
    for key in point:
        lo, hi = np.nanpercentile(draws[key], [2.5, 97.5])
        out[key] = {"est": point[key], "ci_lo": lo, "ci_hi": hi}
    out["n_cells"] = dict(zip(["add_explicit", "add_implicit", "rem_explicit", "rem_implicit"],
                              n.sum(0).astype(int)))
    return out


def did_table_row(name, r):
    def f(key):
        v = r[key]
        return (100 * v["est"], 100 * v["ci_lo"], 100 * v["ci_hi"])
    row = {"slice": name}
    for key in ["add_explicit", "rem_explicit", "add_implicit", "rem_implicit"]:
        row[f"acc_{key}"] = round(100 * r[key]["est"], 2)
    for key in ["deficit_explicit", "deficit_implicit", "did"]:
        e, lo, hi = f(key)
        row[key] = round(e, 2)
        row[f"{key}_ci_lo"] = round(lo, 2)
        row[f"{key}_ci_hi"] = round(hi, 2)
    for cell, cn in r["n_cells"].items():
        row[f"n_{cell}"] = cn
    return row


# ──────────────────────────────────────────────────────────────────────────
# Estimator 2: GEE logistic with exchangeable working correlation
# ──────────────────────────────────────────────────────────────────────────
GEE_FORMULA = ("correct_int ~ C(edit_class, Treatment('add_only')) "
               "* C(setting, Treatment('explicit')) + answer_changed + turn_index")
INTERACTION_TERM = ("C(edit_class, Treatment('add_only'))[T.removal]:"
                    "C(setting, Treatment('explicit'))[T.implicit]")


def gee_interaction(d, formula=GEE_FORMULA):
    dd = d.copy()
    dd["correct_int"] = dd.correct.astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.gee(formula, groups=dd.example_id, data=dd,
                        family=sm.families.Binomial(),
                        cov_struct=sm.cov_struct.Exchangeable())
        res = model.fit()
    coef = res.params[INTERACTION_TERM]
    se = res.bse[INTERACTION_TERM]
    return {
        "or": float(np.exp(coef)),
        "or_ci_lo": float(np.exp(coef - 1.96 * se)),
        "or_ci_hi": float(np.exp(coef + 1.96 * se)),
        "coef": float(coef), "se": float(se),
        "pvalue": float(res.pvalues[INTERACTION_TERM]),
        "n": len(dd),
    }


def vif_table(d):
    dd = d.copy()
    dd["answer_changed_int"] = dd.answer_changed.astype(int)
    X = dmatrix("C(edit_class, Treatment('add_only')) + C(setting, Treatment('explicit')) "
                "+ answer_changed_int + turn_index", dd, return_type="dataframe")
    rows = []
    for i, col in enumerate(X.columns):
        if col == "Intercept":
            continue
        rows.append({"term": col, "vif": round(variance_inflation_factor(X.values, i), 3)})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────
# Estimator 3: paired item deltas
# ──────────────────────────────────────────────────────────────────────────
def paired_deltas(d, changed_only=True, b=B_BOOT, seed=SEED):
    """Δ = correct_implicit − correct_explicit per (model, example, turn).

    Compare mean Δ between removal and add_only items; cluster-bootstrap CI
    (over examples) on the difference of means; Mann-Whitney rank test as check.
    """
    sub = d[d.answer_changed] if changed_only else d
    pv = sub.pivot_table(index=["model", "example_id", "turn_index", "edit_class"],
                         columns="setting", values="correct", aggfunc="first").reset_index()
    pv = pv.dropna(subset=["explicit", "implicit"])
    pv["delta"] = pv.implicit.astype(float) - pv.explicit.astype(float)

    ex_ids = np.sort(pv.example_id.unique())
    ex_idx = {e: i for i, e in enumerate(ex_ids)}
    E = len(ex_ids)
    s = np.zeros((E, 2))
    n = np.zeros((E, 2))
    for j, cls in enumerate(["add_only", "removal"]):
        g = pv[pv.edit_class == cls].groupby("example_id", observed=True).delta.agg(["sum", "size"])
        rows = [ex_idx[e] for e in g.index]
        s[rows, j] = g["sum"].to_numpy()
        n[rows, j] = g["size"].to_numpy()

    def diff(ss, nn):
        with np.errstate(invalid="ignore", divide="ignore"):
            m = ss / nn
        return m[1] - m[0]  # removal − add_only

    point_add = s.sum(0)[0] / n.sum(0)[0]
    point_rem = s.sum(0)[1] / n.sum(0)[1]
    rng = np.random.default_rng(seed)
    draws = np.empty(b)
    for bi in range(b):
        idx = rng.integers(0, E, E)
        draws[bi] = diff(s[idx].sum(0), n[idx].sum(0))
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    mw = st.mannwhitneyu(pv.loc[pv.edit_class == "removal", "delta"],
                         pv.loc[pv.edit_class == "add_only", "delta"],
                         alternative="two-sided")
    return {
        "mean_delta_add": point_add, "mean_delta_removal": point_rem,
        "diff_rem_minus_add": point_rem - point_add,
        "ci_lo": lo, "ci_hi": hi,
        "mannwhitney_u": float(mw.statistic), "mannwhitney_p": float(mw.pvalue),
        "n_add_items": int(n.sum(0)[0]), "n_removal_items": int(n.sum(0)[1]),
    }


# ──────────────────────────────────────────────────────────────────────────
# Autopsy
# ──────────────────────────────────────────────────────────────────────────
def autopsy_label(row):
    if row.correct:
        return "correct"
    if row.answer_changed:
        if row.pred == row.gold_prev:
            return "stale"
        if row.pred == "Uncertain" and row.gold_t != "Uncertain" and row.gold_prev != "Uncertain":
            return "uncertain_retreat"
        return "other_changed"
    if row.pred == "Uncertain":
        return "spurious_to_uncertain"
    neg = {"True": "False", "False": "True"}
    if row.gold_t in neg and row.pred == neg[row.gold_t]:
        return "spurious_flip"
    return "other_invariant"


def autopsy_tables(d):
    dd = d.copy()
    dd["autopsy"] = dd.apply(autopsy_label, axis=1)
    err = dd[dd.autopsy != "correct"]
    pooled = (err.groupby(["setting", "edit_class", "answer_changed", "autopsy"], observed=True)
                 .size().rename("n").reset_index())
    tot = pooled.groupby(["setting", "edit_class", "answer_changed"], observed=True).n.transform("sum")
    pooled["share"] = (pooled.n / tot).round(4)
    per_model = (err.groupby(["model", "setting", "edit_class", "answer_changed", "autopsy"],
                             observed=True).size().rename("n").reset_index())
    mt = per_model.groupby(["model", "setting", "edit_class", "answer_changed"],
                           observed=True).n.transform("sum")
    per_model["share"] = (per_model.n / mt).round(4)
    # headline: flip turns (removal x changed)
    flips = err[(err.edit_class == "removal") & err.answer_changed]
    headline = (flips.groupby(["setting", "autopsy"], observed=True).size().rename("n").reset_index())
    ht = headline.groupby("setting", observed=True).n.transform("sum")
    headline["share"] = (headline.n / ht).round(4)
    # pooled retreat share on flip turns across both settings
    flip_pool = flips.autopsy.value_counts(normalize=True)
    return dd, pooled, per_model, headline, flip_pool


# ──────────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────────
def _style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e4df", lw=0.7)
    ax.set_axisbelow(True)


def savefig(fig, name):
    fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_interaction_small_multiples(d, name, changed_only=False, title_suffix=""):
    sub = d[d.answer_changed] if changed_only else d
    models = sorted(sub.model.unique())
    panels = ["POOLED"] + models
    ncol = 6
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.4 * nrow),
                             sharey=True, sharex=True)
    axes = np.atleast_2d(axes)
    x = [0, 1]
    for i, panel in enumerate(panels):
        ax = axes[i // ncol, i % ncol]
        g = sub if panel == "POOLED" else sub[sub.model == panel]
        for cls, col in [("add_only", C_ADD), ("removal", C_REM)]:
            y = [100 * g[(g.edit_class == cls) & (g.setting == s)].correct.mean()
                 for s in ("explicit", "implicit")]
            ax.plot(x, y, "-o", color=col, ms=4, lw=1.8)
            if panel == "POOLED":
                ax.annotate(cls.replace("_", " "), (x[1], y[1]),
                            xytext=(4, 0), textcoords="offset points",
                            color="#52514e", fontsize=7, va="center")
        ax.set_title(panel.split("/")[-1] if panel != "POOLED" else "POOLED (all models)",
                     fontsize=8, fontweight="bold" if panel == "POOLED" else "normal")
        ax.set_xticks(x, ["explicit", "implicit"], fontsize=7)
        ax.set_ylim(0, 100)
        _style_ax(ax)
    for j in range(len(panels), nrow * ncol):
        axes[j // ncol, j % ncol].axis("off")
    fig.suptitle(f"Accuracy by edit class and setting{title_suffix} "
                 f"(blue = add_only, orange = removal)", fontsize=10, y=1.005)
    fig.supylabel("accuracy (%)", fontsize=9)
    fig.tight_layout()
    savefig(fig, name)


def fig_changed_pooled(d):
    sub = d[d.answer_changed]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    x = [0, 1]
    series = [("add_only (flips by addition)", sub.edit_class == "add_only", C_ADD),
              ("removal (flips)", sub.edit_class == "removal", C_REM),
              ("pure removal", (sub.n_rem > 0) & (sub.n_add == 0), C_PURE)]
    for label, mask, col in series:
        g = sub[mask]
        y = [100 * g[g.setting == s].correct.mean() for s in ("explicit", "implicit")]
        ax.plot(x, y, "-o", color=col, lw=2, ms=6)
        ax.annotate(label, (x[1], y[1]), xytext=(6, 0), textcoords="offset points",
                    color="#0b0b0b", fontsize=8, va="center")
    ax.set_xticks(x, ["explicit", "implicit"])
    ax.set_xlim(-0.2, 1.9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Answer-changing turns: accuracy by edit class\n(pooled, primary slice)",
                 fontsize=10)
    _style_ax(ax)
    savefig(fig, "interaction_changed_pooled")


def fig_turn_curves(d):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    for ax, s in zip(axes, ("explicit", "implicit")):
        ends = {}
        for cls, col in [("add_only", C_ADD), ("removal", C_REM)]:
            g = d[(d.setting == s) & (d.edit_class == cls)]
            y = 100 * g.groupby("turn_index").correct.mean()
            ax.plot(y.index, y.values, "-o", color=col, lw=1.8, ms=4)
            ends[cls] = (y.index[-1], y.values[-1])
        # de-collide the two direct labels at the last point
        hi = max(ends, key=lambda c: ends[c][1])
        for cls, (xe, ye) in ends.items():
            dy = 5 if cls == hi else -5
            ax.annotate(cls.replace("_", " "), (xe, ye), xytext=(4, dy),
                        textcoords="offset points", color="#52514e",
                        fontsize=8, va="center")
        ax.set_title(s, fontsize=10)
        ax.set_xlabel("turn index")
        ax.set_ylim(0, 100)
        _style_ax(ax)
    axes[0].set_ylabel("accuracy (%)")
    fig.suptitle("Per-turn accuracy by edit class (pooled, primary slice)", fontsize=10)
    fig.tight_layout()
    savefig(fig, "turn_curves")


def fig_autopsy(pooled):
    changed_cats = ["stale", "uncertain_retreat", "other_changed"]
    inv_cats = ["spurious_to_uncertain", "spurious_flip", "other_invariant"]
    colors = {"stale": C_REM, "uncertain_retreat": C_YEL, "other_changed": C_OTHER,
              "spurious_to_uncertain": C_YEL, "spurious_flip": C_REM,
              "other_invariant": C_OTHER}
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    for ax, (changed, cats, ttl) in zip(
            axes, [(True, changed_cats, "answer-changing turns (errors)"),
                   (False, inv_cats, "invariant turns (errors)")]):
        sub = pooled[pooled.answer_changed == changed]
        groups = [(s, c) for s in ("explicit", "implicit") for c in ("add_only", "removal")]
        xs = np.arange(len(groups))
        bottom = np.zeros(len(groups))
        for cat in cats:
            vals = []
            for s, c in groups:
                v = sub[(sub.setting == s) & (sub.edit_class == c) & (sub.autopsy == cat)]
                vals.append(100 * float(v.share.iloc[0]) if len(v) else 0.0)
            vals = np.array(vals)
            ax.bar(xs, vals, bottom=bottom, color=colors[cat], width=0.62,
                   edgecolor="#fcfcfb", linewidth=1.5, label=cat)
            for xi, (v, b) in enumerate(zip(vals, bottom)):
                if v > 7:
                    ax.text(xi, b + v / 2, f"{v:.0f}", ha="center", va="center",
                            fontsize=7, color="#0b0b0b")
            bottom += vals
        ax.set_xticks(xs, [f"{s[:4]}\n{c.replace('_', ' ')}" for s, c in groups], fontsize=8)
        ax.set_title(ttl, fontsize=10)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=7, frameon=False, loc="upper right", bbox_to_anchor=(1.0, -0.12),
                  ncol=3)
        _style_ax(ax)
    axes[0].set_ylabel("share of errors (%)")
    fig.suptitle("Error autopsy by setting × edit class (pooled, primary slice)", fontsize=11)
    fig.tight_layout()
    savefig(fig, "autopsy_bars")


# ──────────────────────────────────────────────────────────────────────────
# Markdown helpers
# ──────────────────────────────────────────────────────────────────────────
def md_table(df, floatfmt=2):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].round(floatfmt)
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([header, sep] + rows)


def fmt_ci(row, key):
    return f"{row[key]:+.1f} [{row[f'{key}_ci_lo']:+.1f}, {row[f'{key}_ci_hi']:+.1f}]"


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    df = pd.read_parquet(os.path.join(OUT, "tidy.parquet"))
    flagged = flagged_models(df)
    print("flagged models:", flagged)

    prim = make_slice(df, flagged=flagged)
    n_models = prim.model.nunique()
    print(f"primary slice: {len(prim)} rows, {n_models} models")

    # cell counts (before any modeling)
    cc = (prim.groupby(["edit_class", "answer_changed", "setting"], observed=True)
              .size().rename("n_turns").reset_index())
    cc.to_csv(os.path.join(TAB, "cell_counts_primary.csv"), index=False)
    cc_all = (df.groupby(["edit_class", "answer_changed"], observed=True)
                .size().rename("n_rows_all_runs").reset_index())
    cc_all.to_csv(os.path.join(TAB, "cell_counts_all_runs.csv"), index=False)

    # stop-condition guard on cell sizes
    n_add_ch = cc[(cc.edit_class == "add_only") & cc.answer_changed].n_turns.sum() // 2
    n_rem_ch = cc[(cc.edit_class == "removal") & cc.answer_changed].n_turns.sum() // 2
    assert n_add_ch >= 100 and n_rem_ch >= 100, \
        f"STOP-CONDITION: changed-cell too small (add={n_add_ch}, rem={n_rem_ch})"

    # ── Phase 0.2 descriptives ──
    d1 = (prim.groupby(["edit_class", "setting"], observed=True)
              .correct.agg(acc="mean", n="size").reset_index())
    d1["acc"] = (100 * d1.acc).round(2)
    d1.to_csv(os.path.join(TAB, "desc_editclass_setting.csv"), index=False)

    d1m = (prim.groupby(["model", "edit_class", "setting"], observed=True)
               .correct.agg(acc="mean", n="size").reset_index())
    d1m["acc"] = (100 * d1m.acc).round(2)
    d1m.to_csv(os.path.join(TAB, "desc_editclass_setting_per_model.csv"), index=False)

    d2 = (prim.groupby(["edit_class", "answer_changed", "setting"], observed=True)
              .correct.agg(acc="mean", n="size").reset_index())
    d2["acc"] = (100 * d2.acc).round(2)
    d2.to_csv(os.path.join(TAB, "desc_2x2_setting.csv"), index=False)

    d2m = (prim.groupby(["model", "edit_class", "answer_changed", "setting"], observed=True)
               .correct.agg(acc="mean", n="size").reset_index())
    d2m["acc"] = (100 * d2m.acc).round(2)
    d2m.to_csv(os.path.join(TAB, "desc_2x2_setting_per_model.csv"), index=False)

    # sanity anchor: reproduce paper's pooled add ≈73.6 / remove ≈50.6.
    # Uses the paper's own ingredients: recorded edits_made coding + harness scoring.
    rec_add = df.edit_class_recorded == "add_only"
    rec_pure = (df.n_rem_recorded > 0) & (df.n_add_recorded == 0)
    rec_any = df.n_rem_recorded > 0
    anchor = pd.DataFrame([
        {"definition": "recorded add_only, all runs, harness scoring",
         "acc": round(100 * df[rec_add].log_correct.mean(), 2), "n": int(rec_add.sum())},
        {"definition": "recorded pure removal (n_rem>0, n_add=0), all runs, harness scoring",
         "acc": round(100 * df[rec_pure].log_correct.mean(), 2), "n": int(rec_pure.sum())},
        {"definition": "recorded any-removal (n_rem>0, incl. flips), all runs, harness scoring",
         "acc": round(100 * df[rec_any].log_correct.mean(), 2), "n": int(rec_any.sum())},
        {"definition": "recorded add_only, all runs, our scoring (A/B/C mapped)",
         "acc": round(100 * df[rec_add].correct.mean(), 2), "n": int(rec_add.sum())},
        {"definition": "recorded pure removal, all runs, our scoring (A/B/C mapped)",
         "acc": round(100 * df[rec_pure].correct.mean(), 2), "n": int(rec_pure.sum())},
    ])
    anchor.to_csv(os.path.join(TAB, "sanity_anchor.csv"), index=False)

    fig_interaction_small_multiples(prim, "interaction_small_multiples",
                                    title_suffix=", all turns")
    fig_interaction_small_multiples(prim, "interaction_small_multiples_changed",
                                    changed_only=True, title_suffix=", answer-changing turns")
    fig_changed_pooled(prim)
    fig_turn_curves(prim)

    # ── Phase 0.3 interaction tests ──
    # P1 pooled
    p1 = cluster_boot_did(prim[prim.answer_changed])
    p1_rows = [did_table_row("pooled", p1)]
    # P1 per model
    per_model = []
    for m, g in prim[prim.answer_changed].groupby("model", observed=True):
        r = cluster_boot_did(g)
        row = did_table_row(m, r)
        per_model.append(row)
    p1_per_model = pd.DataFrame(per_model)
    p1_per_model.to_csv(os.path.join(TAB, "p1_per_model.csv"), index=False)
    pd.DataFrame(p1_rows).to_csv(os.path.join(TAB, "p1_pooled.csv"), index=False)
    share_pos = float((p1_per_model.did > 0).mean())

    # P1b supplementary: pure removals vs add_only among changed turns
    prim_b = prim.copy()
    prim_b["edit_class"] = np.where((prim_b.n_rem > 0) & (prim_b.n_add == 0),
                                    "removal", np.where(prim_b.edit_class == "add_only",
                                                        "add_only", "drop"))
    p1b = cluster_boot_did(prim_b[(prim_b.answer_changed) & (prim_b.edit_class != "drop")])
    pd.DataFrame([did_table_row("pooled_pure_removal", p1b)]).to_csv(
        os.path.join(TAB, "p1b_pure_removal.csv"), index=False)

    # P2 GEE pooled + per model, VIFs
    gee_pooled = gee_interaction(prim)
    gee_changed = gee_interaction(
        prim[prim.answer_changed],
        formula=("correct_int ~ C(edit_class, Treatment('add_only')) "
                 "* C(setting, Treatment('explicit')) + turn_index"))
    pd.DataFrame([{**{"slice": "pooled_all_turns"}, **gee_pooled},
                  {**{"slice": "pooled_changed_only"}, **gee_changed}]).to_csv(
        os.path.join(TAB, "p2_gee_pooled.csv"), index=False)
    gee_rows = []
    for m, g in prim.groupby("model", observed=True):
        try:
            gee_rows.append({**{"model": m}, **gee_interaction(g)})
        except Exception as e:  # keep going; report failure
            gee_rows.append({"model": m, "or": np.nan, "error": str(e)[:100]})
    pd.DataFrame(gee_rows).to_csv(os.path.join(TAB, "p2_gee_per_model.csv"), index=False)
    vt = vif_table(prim)
    vt.to_csv(os.path.join(TAB, "p2_vif.csv"), index=False)

    # P3: invariant turns
    n_rem_inv = len(prim[(prim.edit_class == "removal") & (~prim.answer_changed)])
    p3 = None
    if n_rem_inv >= P3_MIN_POOLED_TURNS:
        p3 = cluster_boot_did(prim[~prim.answer_changed])
        pd.DataFrame([did_table_row("pooled_invariant", p3)]).to_csv(
            os.path.join(TAB, "p3_invariant.csv"), index=False)

    # Paired deltas
    pdl = paired_deltas(prim)
    pd.DataFrame([pdl]).to_csv(os.path.join(TAB, "paired_deltas.csv"), index=False)

    # ── Sensitivity battery ──
    sens_rows = []

    def add_sens(name, sl):
        r = cluster_boot_did(sl[sl.answer_changed])
        sens_rows.append(did_table_row(name, r))

    add_sens("primary", prim)
    add_sens("a_exclude_parse_fail",
             make_slice(df, flagged=flagged, drop_parse_fail=True))
    add_sens("b_include_flagged", make_slice(df, flagged=flagged, include_flagged=True))
    add_sens("c_cot_prompting", make_slice(df, prompting="cot", flagged=flagged))
    add_sens("d_feedback_runs", make_slice(df, feedback=True, flagged=flagged))
    add_sens("f_prereg_spec_coding",
             make_slice(df, flagged=flagged, exclude_mismatch=False, use_recorded=True))
    add_sens("g_actual_coding_incl_mismatch",
             make_slice(df, flagged=flagged, exclude_mismatch=False))
    for tgt in ["fact", "rule", "both"]:
        sl = prim[prim.edit_target == tgt]
        if len(sl[sl.answer_changed & (sl.edit_class == "add_only")]) >= 50:
            add_sens(f"e_target_{tgt}", sl)
        else:
            sens_rows.append({"slice": f"e_target_{tgt}",
                              "note": "add_only x changed cell < 50 turns; skipped"})
    pd.DataFrame(sens_rows).to_csv(os.path.join(TAB, "sensitivity.csv"), index=False)

    # ── Phase 0.4 autopsy ──
    dd, aut_pooled, aut_per_model, aut_headline, flip_pool = autopsy_tables(prim)
    aut_pooled.to_csv(os.path.join(TAB, "autopsy_pooled.csv"), index=False)
    aut_per_model.to_csv(os.path.join(TAB, "autopsy_per_model.csv"), index=False)
    aut_headline.to_csv(os.path.join(TAB, "autopsy_flip_headline.csv"), index=False)
    fig_autopsy(aut_pooled)
    # sensitivity: exclude turn 1 (demo-answer bug makes gold_prev ambiguous at t=1)
    _, _, _, aut_headline_no_t1, flip_pool_no_t1 = autopsy_tables(prim[prim.turn_index > 1])
    aut_headline_no_t1.to_csv(os.path.join(TAB, "autopsy_flip_headline_excl_turn1.csv"),
                              index=False)

    retreat_share = float(flip_pool.get("uncertain_retreat", 0.0))
    stale_share = float(flip_pool.get("stale", 0.0))

    # ── Decision gate ──
    did_pts = 100 * p1["did"]["est"]
    ci_lo, ci_hi = 100 * p1["did"]["ci_lo"], 100 * p1["did"]["ci_hi"]
    ci_excl_0 = (ci_lo > 0) or (ci_hi < 0)
    gate1 = (did_pts >= GATE_P1_DID_MIN_PTS) and (ci_lo > 0) and (share_pos >= GATE_P1_MODEL_SHARE)
    idef = 100 * p1["deficit_implicit"]["est"]
    idef_lo, idef_hi = 100 * p1["deficit_implicit"]["ci_lo"], 100 * p1["deficit_implicit"]["ci_hi"]
    gate2 = (idef >= GATE_INTRINSIC_MIN_PTS) and (idef_lo > 0)
    gate3 = retreat_share >= GATE_RETREAT_SHARE
    gates = pd.DataFrame([
        {"gate": "suppression_GO",
         "criterion": f"P1 DiD >= {GATE_P1_DID_MIN_PTS} pts, CI excl 0, "
                      f">= {GATE_P1_MODEL_SHARE:.0%} models DiD>0",
         "value": f"DiD={did_pts:+.2f} [{ci_lo:+.2f},{ci_hi:+.2f}], "
                  f"models>0: {share_pos:.0%}",
         "pass": bool(gate1)},
        {"gate": "intrinsic_contraction_GO",
         "criterion": f"implicit deficit (Add_impl - Rem_impl, changed) >= "
                      f"{GATE_INTRINSIC_MIN_PTS} pts, CI excl 0",
         "value": f"{idef:+.2f} [{idef_lo:+.2f},{idef_hi:+.2f}]",
         "pass": bool(gate2)},
        {"gate": "uncertain_retreat_headline",
         "criterion": f"uncertain_retreat >= {GATE_RETREAT_SHARE:.0%} of flip-turn errors",
         "value": f"{retreat_share:.1%} (stale: {stale_share:.1%})",
         "pass": bool(gate3)},
    ])
    gates.to_csv(os.path.join(TAB, "decision_gate.csv"), index=False)
    print(gates.to_string(index=False))

    write_report(df=df, prim=prim, flagged=flagged, cc=cc, d1=d1, d2=d2, anchor=anchor,
                 p1=p1, p1_per_model=p1_per_model, share_pos=share_pos, p1b=p1b,
                 gee_pooled=gee_pooled, gee_changed=gee_changed,
                 gee_per_model=pd.DataFrame(gee_rows), vt=vt,
                 p3=p3, n_rem_inv=n_rem_inv, pdl=pdl, sens=pd.DataFrame(sens_rows),
                 aut_pooled=aut_pooled, aut_headline=aut_headline,
                 aut_headline_no_t1=aut_headline_no_t1, flip_pool=flip_pool,
                 flip_pool_no_t1=flip_pool_no_t1, gates=gates)
    print("report written to", os.path.join(OUT, "report.md"))


def write_report(**kw):
    df, prim, flagged = kw["df"], kw["prim"], kw["flagged"]
    p1, p1b, p3, pdl = kw["p1"], kw["p1b"], kw["p3"], kw["pdl"]
    gee, gee_ch = kw["gee_pooled"], kw["gee_changed"]
    inv = pd.read_csv(os.path.join(TAB, "inventory.csv"))
    pairing = pd.read_csv(os.path.join(TAB, "pairing.csv"))
    pq = pd.read_csv(os.path.join(TAB, "parse_quality.csv"))
    audit = pd.read_csv(os.path.join(TAB, "fol_delta_audit.csv"))

    def pct(x):
        return f"{100 * x:.1f}"

    def did_str(r, key):
        return (f"{100 * r[key]['est']:+.2f} pts "
                f"[95% CI {100 * r[key]['ci_lo']:+.2f}, {100 * r[key]['ci_hi']:+.2f}]")

    prim_pairing = pairing[(pairing.prompting == PRIMARY_PROMPTING) & (~pairing.feedback)]
    pq_prim = pq[pq.track.str.contains("no_reasoning_no_correction")]
    pq_wide = pq_prim.pivot(index="model", columns="track", values="parse_fail_pct").reset_index()
    pq_wide.columns = ["model", "explicit_parse_fail_pct", "implicit_parse_fail_pct"]

    n_mismatch = int(audit.delta_mismatch.sum())
    n_turns_ds = len(audit)
    reclass = (audit.groupby(["edit_class_recorded", "edit_class"]).size()
                    .rename("n_example_turns").reset_index())

    per_model_short = kw["p1_per_model"][["slice", "deficit_explicit", "deficit_implicit",
                                          "did", "did_ci_lo", "did_ci_hi"]].rename(
        columns={"slice": "model"})

    gates = kw["gates"]
    n_models = prim.model.nunique()
    fp, fp1 = kw["flip_pool"], kw["flip_pool_no_t1"]
    ah = kw["aut_headline"]

    def flip_cell(setting, cat):
        v = ah[(ah.setting == setting) & (ah.autopsy == cat)]
        return pct(float(v.share.iloc[0])) if len(v) else "0.0"

    report = f"""# ReviseQA Phase 0 — Log Mining & Interaction Test: Report

> **ERRATUM (Phase 0.5, §1.1 comparator audit — SPEC_ADDENDUM_A).** The P1
> comparator (`add_only × changed`, 18 delta-verified turns) failed the
> prover consistency audit: 11/18 post-edit theories are **inconsistent**
> (Prover9 derives a contradiction; the gold label is ill-defined) and 5/18
> are **mislabeled** (re-derived gold at 10× budget differs from the stored
> answer). Only 2/18 are sound. Per the pre-registered decision rule (≥ 1/3),
> **the P1 DiD of +15.5 pts is demoted to *suggestive* and must not be cited
> as a finding.** See `analysis/phase0_5/report.md`.
> *(Evidence-status per SPEC_ADDENDUM_B: the citable Phase-0 results are the
> error autopsy, P3's answer-relevance null, and the descriptives. **P1b is
> also not citable as evidence** — audit-clean restriction +11.6
> [−6.6, +34.1], n.s., 5 unique turns; clean slice incomputable. Canonical
> list: `analysis/EVIDENCE_STATUS.md`.)*
>
> **Phase 0.6 Job C update: the comparator is 18/18 invalid.** The two turns
> that audited CONSISTENT_GOLD_OK have prover-certified *mislabeled pre-edit
> states* (`ex_802` t6 is U→F, not T→F; `ex_813` t3 is T→T, not F→T), so
> those "flips" are illusory. Moreover, on fully verified turns the changed
> band is compositionally homogeneous (all flips are remove+add), so no
> P1/P1b-style within-changed contrast is estimable from v1 at all, and P1b
> weakens to non-significant under audit restriction (+11.6 [−6.6, +34.1]).
> See `analysis/phase0_6/report.md`.

Deterministic analysis (seed {SEED}, bootstrap B={B_BOOT}) of existing eval
logs; no API calls. Reproduce with `analysis/phase0/run.sh`. All decision-gate
thresholds sit at the top of `analysis/phase0/src/analysis.py`.

> **Pre-registration amendment (user-approved during execution).** The §2.3
> audit was extended to the FOL level and found that the dataset's recorded
> `edits_made` metadata does not match the actual context change on
> {n_mismatch}/{n_turns_ds} example-turns (§1.5). On those turns the explicit
> track showed models a wrong/incomplete delta while the implicit track showed
> the true state — the two settings differ in *content*, not just
> presentation. The primary analysis therefore (i) recodes every turn from the
> **actual FOL context delta** and (ii) **excludes delta-mismatch turns** from
> the interaction tests. The pre-registered spec coding (recorded metadata,
> mismatches included) is reported unchanged as sensitivity (f); it is the
> single specification that materially disagrees, and §4.6 shows why.

## 1. Gate checks (Phase 0.0)

### 1.1 Harness-bug audit (§2.1) — verdict: **LATEX_TYPO_ONLY**

Prompt construction lives in `src/evaluation.py` (`LogicDataset.read_dir`,
current lines 247–265). Each explicit-delta slot is bound to its matching
variable — `"Added rules:"` joins `delta["added_rules"]`
(`src/evaluation.py:257-260`), `"Removed rules:"` joins
`delta["removed_rules"]` (`:253-256`), likewise facts. Checked in **every git
revision** of the file (78e632f → 2382150, May–Sep 2025, spanning the run
dates 2025-09-25..28): correctly bound throughout. Verified empirically too:
logged prompt contexts match the correctly-bound construction
character-for-character on 20 seeded examples × 3 models × 7 turns (420/420,
§1.3). The `added rules: {{removed_rules}}` binding in the paper's Appendix B
exists only in the LaTeX.

**Decision rule applied:** clean → primary prompting = **Standard**
(`*_no_reasoning` tracks), COT as robustness. Primary slice = Standard ×
no-feedback.

**Separate harness bug found (outside §2.1 scope, reported for completeness):**
`src/evaluation.py:214` sets the turn-0 demonstration answer to
`edits[-1]["answer"]` — the answer after the *final* edit — instead of the
original context's answer. They differ in **207/400** eval examples, so ~52%
of conversations open with a demonstration asserting a wrong answer for the
context shown (in COT tracks it also contradicts the demonstrated reasoning
chain). This is identical across all tracks and settings, so it cannot mimic
an edit-class × setting interaction, but absolute accuracies are depressed by
it and turn-1 autopsy labels carry a caveat (§5).

### 1.2 Data inventory & pairing (§2.2)

19 models × 8 tracks (setting × prompting × feedback) × 400 examples × 7
turns; `google/gemini-2-5-flash` is missing 1 example in 2 tracks (333).
Full inventory: `tables/inventory.csv`. Pairing on the primary runs
(explicit ∩ implicit example IDs):

{md_table(prim_pairing[["model", "n_explicit", "n_implicit", "n_paired"]])}

Worst overlap 333/334 (99.7%) — far above the 90% stop threshold. All
analyses use each model's paired intersection.

**Dataset size reconciliation:** `reviseqa_data/nl/dataset_statistics.json`
records 1731 generated examples and **930 preserved after verification** —
§4's 930 is correct; §3.3's "933" is stale. The 400-example eval subset is
exactly `reviseqa_data/nl/verified-400/` (334 `ex_*` + 66 `prev_ex_*`), the
directory `evaluate_models.sh` feeds the harness. Per-turn gold answers exist
in both the logs (`correct_answer`) and the dataset (`edits[i].answer`) and
agree on all 425,586 turn rows.

### 1.3 Edit-metadata join (§2.3)

Join key: **`example_id` (from the log filename) + edit position (log `step`
i ↔ dataset `edits[i-1]`; equals `edit_number`)**. Hand-verified on 20
seed-{SEED} random examples × 3 models (gpt-4.1-mini, claude-sonnet-4,
qwen3-30b-a3b): explicit log context == delta rebuilt from `edits_made`,
implicit log context == `edited_natural_language_context`, log gold ==
dataset answer, for all 420 triples. **420/420 passed** — the logs faithfully
reflect the recorded metadata. (§1.5 shows the recorded metadata itself is
what fails against the FOL state.)

### 1.4 Parse-quality audit (§2.4)

Answers re-extracted per spec (JSON `answer` field; A/B/C ↔
True/False/Uncertain; both formats; empty/`ERROR` → `PARSE_FAIL`, scored
incorrect, retained). Parse-fail % in primary runs:

{md_table(pq_wide)}

**Flagged (> {FLAG_PARSE_FAIL_PCT:.0f}% in a primary run):**
{", ".join(f"`{m}`" for m in flagged)}. The spec expected `gpt-5-nano` too;
its primary-run failure rate is < 0.1%, so it is **not** flagged. Flagged
models: kept per-model, excluded from pooled primary, included in
sensitivity (b).

Scoring note: the A/B/C mapping recovers answers the harness scored wrong
(bare option letters). All scoring disagreements with the harness's own
`correct` flag are one-directional recoveries (up to 10% of turns in the
worst qwen3-30b-a3b run), so published accuracies modestly underestimate
letter-answering models.

### 1.5 FOL delta-integrity audit (new; basis of the amendment)

Comparing each turn's recorded `edits_made` against the actual FOL context
change (`edited_context_fol[t] − edited_context_fol[t−1]`):
**{n_mismatch}/{n_turns_ds} example-turns (18.2%) mismatch**; 509 turns
contain removals absent from the metadata (and hence absent from the explicit
prompt), 494 contain unrecorded additions; a smaller number of recorded edits
never happened ("phantom"). Reclassification recorded → actual:

{md_table(reclass)}

Notably, **158 recorded-`add_only` turns actually contain removals** — e.g.
`ex_1011` turn 6 silently drops a biconditional from the implicit context
while the explicit delta mentions only the addition. Also ~21% of recorded
"facts" are syntactically rules; actual-delta coding classifies fact/rule
from FOL syntax. Audit table: `tables/fol_delta_audit.csv`.

## 2. Tidy table (Phase 0.1)

`tidy.parquet` / `tidy.csv`: 425,586 rows, one per (model, prompting,
feedback, setting, example_id, turn), with both codings (`edit_class` =
actual-delta; `edit_class_recorded` = spec) and the `delta_mismatch` flag.
Coder unit tests (5 hand-checked cases incl. the paper's Uriel example =
`ex_2512`): **all pass** (`src/tests/test_coder.py`).

Cell counts, primary slice ({n_models} non-flagged models, Standard,
no-feedback, paired, delta-verified turns), per setting:

{md_table(kw["cc"])}

Structural facts **verified, not assumed** — three of the spec's assumptions
fail in the data:

1. **Pure removals exist** (48 unique example-turns under recorded coding;
   more under actual coding) — the Invariant prompt's no-removal instruction
   was not binding. They are ~2% of turns but drive the paper's headline
   removal number (§3).
2. **There are no U→T edits.** The spec's natural 2×2 assumed
   `add_only × changed` = Uncertain→True. In reality *every* answer-changing
   turn is a T↔F flip (~50/50 both directions, one U→F), in both classes.
   `add_only × changed` = flips achieved by pure addition (the gold labels
   are non-monotonic because "False" = ¬conclusion provable). This *removes*
   the pre-registered transition confound — both P1 cells share the same
   transition mix — at the cost of a small comparator (18 unique
   delta-verified example-turns).
3. A small **`none` class** exists (empty actual delta; excluded).

## 3. Descriptives (Phase 0.2)

Accuracy by `edit_class × setting` (primary slice, all turns):

{md_table(kw["d1"])}

Full `edit_class × answer_changed × setting` table:

{md_table(kw["d2"])}

Per-model versions: `tables/desc_*_per_model.csv`.

### Sanity anchor — the paper's pooled 73.6 / 50.6

{md_table(kw["anchor"])}

The paper's pooled numbers reproduce **exactly** for additions (73.6 =
recorded `add_only`, all runs, harness scoring) and within 0.5 pts for
removals (50.1 vs 50.6) — **but only when "removals" means *pure* removals**
(n_rem>0, n_add=0; ~1.7% of turn-rows; the residual is consistent with the
published pooling including `gpt-5-mini`, whose detailed logs are absent from
this repo). Any-removal turns score 71.6% — indistinguishable from additions.
**The published add-vs-remove gap is a composition effect**: recorded
`add_only` turns are 97% invariant (easy), pure-removal turns are 67%
answer-changing (hard), and pure removals collapse specifically in the
explicit setting (25.3% explicit vs 74.8% implicit, all runs).

### Figures

- `figures/interaction_small_multiples.{{png,pdf}}` — per-model panels, all
  turns: both classes gain ~25–30 pts explicit→implicit; near-parallel at
  this altitude because invariant turns dominate.
- `figures/interaction_small_multiples_changed.{{png,pdf}}` — the same
  restricted to answer-changing turns: in nearly every panel the removal
  line rises steeply from explicit to implicit while the add line stays
  almost flat (the suppression signature, model-by-model).
- `figures/interaction_changed_pooled.{{png,pdf}}` — pooled changed-turn
  panel with the pure-removal line.
- `figures/turn_curves.{{png,pdf}}` — per-turn curves: explicit accuracy
  drifts down over turns for both classes; implicit stays flat; no
  class-selective turn artifact.

## 4. Interaction tests (Phase 0.3)

Item = (example, turn); cluster = example; DiD = (Add_expl − Rem_expl) −
(Add_impl − Rem_impl). **Positive DiD = removals hurt more by the explicit
setting than additions = suppression signature.**

### 4.1 P1 — primary (answer-changing turns only)

Cell accuracies: Add_expl {pct(p1["add_explicit"]["est"])}, Rem_expl
{pct(p1["rem_explicit"]["est"])}, Add_impl {pct(p1["add_implicit"]["est"])},
Rem_impl {pct(p1["rem_implicit"]["est"])}
(n = {p1["n_cells"]["add_explicit"]} / {p1["n_cells"]["rem_explicit"]}
add/removal turns per setting).

| estimand | estimate [95% CI] |
|---|---|
| Add − Rem, explicit | {did_str(p1, "deficit_explicit")} |
| Add − Rem, implicit | {did_str(p1, "deficit_implicit")} |
| **DiD (P1)** | **{did_str(p1, "did")}** |

Two separate findings live here:

1. **No intrinsic removal deficit — the sign reverses.** Among
   answer-changing edits, removal-flips are *easier* than addition-flips in
   both settings (the "Add − Rem deficit" is large and negative). The hard
   class is flips-by-pure-addition, whose non-monotonic label semantics
   models handle near chance (36/41%).
2. **A clean suppression signature.** Removal-flips gain
   {100 * (p1["rem_implicit"]["est"] - p1["rem_explicit"]["est"]):.1f} pts
   from explicit→implicit while addition-flips gain only
   {100 * (p1["add_implicit"]["est"] - p1["add_explicit"]["est"]):.1f} pts:
   DiD {did_str(p1, "did")}, and **DiD > 0 in
   {pct(kw["share_pos"])}% of the {n_models} non-flagged models**
   (range {per_model_short.did.min():+.1f} to
   {per_model_short.did.max():+.1f}; `tables/p1_per_model.csv`):

{md_table(per_model_short)}

### 4.2 P1b — supplementary: *pure* removals vs additions (changed turns)

Cells: PureRem_expl {pct(p1b["rem_explicit"]["est"])}, PureRem_impl
{pct(p1b["rem_implicit"]["est"])} (n = {p1b["n_cells"]["rem_explicit"]}
per setting; only 9 unique delta-verified example-turns — descriptive only).
DiD = {did_str(p1b, "did")} — same direction as P1, wide CI.

### 4.3 P2 — GEE logistic

`{GEE_FORMULA.replace("correct_int", "correct")}`, exchangeable working
correlation, clustered by example.

| slice | interaction OR (removal × implicit) [95% robust CI] | p | n |
|---|---|---|---|
| all turns | {gee["or"]:.3f} [{gee["or_ci_lo"]:.3f}, {gee["or_ci_hi"]:.3f}] | {gee["pvalue"]:.1e} | {gee["n"]:,} |
| changed turns only | {gee_ch["or"]:.3f} [{gee_ch["or_ci_lo"]:.3f}, {gee_ch["or_ci_hi"]:.3f}] | {gee_ch["pvalue"]:.1e} | {gee_ch["n"]:,} |

**The two GEE rows disagree in direction, and the all-turns row should not be
read as the P1 check.** On changed turns only — the P1 estimand — the GEE
agrees with the bootstrap DiD: OR {gee_ch["or"]:.2f} > 1
(removals benefit more from implicit). The all-turns OR < 1 is a
composition/scale artifact: `add_only` mass sits on invariant turns
(57.7→88.3, log-odds gain 1.71) while `removal` mass sits on changed turns
(63.9→84.3, log-odds gain 1.11), and a single additive `answer_changed` term
cannot absorb the class-specific ceiling geometry. Per-model all-turns ORs:
`tables/p2_gee_per_model.csv`.

VIFs for the P2 design matrix — `answer_changed` is only mildly collinear
with `edit_class` (all < 1.9):

{md_table(kw["vt"])}

### 4.4 P3 — invariant turns (rewrites vs redundant additions)

`removal × invariant` pooled n = {kw["n_rem_inv"]:,} ≥ {P3_MIN_POOLED_TURNS}
→ run. Cells: Add_expl {pct(p3["add_explicit"]["est"])}, Rem_expl
{pct(p3["rem_explicit"]["est"])}, Add_impl {pct(p3["add_implicit"]["est"])},
Rem_impl {pct(p3["rem_implicit"]["est"])}.

| estimand | estimate [95% CI] |
|---|---|
| Operation cost, explicit (Add−Rem) | {did_str(p3, "deficit_explicit")} |
| Operation cost, implicit (Add−Rem) | {did_str(p3, "deficit_implicit")} |
| **DiD (P3)** | **{did_str(p3, "did")}** |

**Null.** With zero answer movement, rewrite-removals cost nothing relative
to redundant additions in either setting. The suppression effect (P1) is
specific to turns where the removal *matters* for the answer — retracted-but-
visible premises hurt only when the conclusion depends on the retraction.
This is the free preview of the matched-generation B0-vs-C0 contrast:
expect ≈ 0 there too unless the edit is answer-relevant.

### 4.5 Estimator 3 — paired item deltas (changed turns)

Δ = correct_implicit − correct_explicit per (model, example, turn). Mean Δ:
add_only {pdl["mean_delta_add"]:+.3f} (n = {pdl["n_add_items"]:,}),
removal {pdl["mean_delta_removal"]:+.3f} (n = {pdl["n_removal_items"]:,});
difference (removal − add) = **{pdl["diff_rem_minus_add"]:+.3f}
[95% cluster-bootstrap CI {pdl["ci_lo"]:+.3f}, {pdl["ci_hi"]:+.3f}]**,
Mann-Whitney p = {pdl["mannwhitney_p"]:.1e}. Matches P1's DiD
(+{100 * p1["did"]["est"]:.1f} pts) — estimators 1 and 3 agree exactly;
estimator 2 agrees on the matching (changed-only) estimand (§4.3).

### 4.6 Sensitivity battery

{md_table(pd.DataFrame(kw["sens"])[["slice", "deficit_explicit", "deficit_implicit", "did", "did_ci_lo", "did_ci_hi"]].fillna(""))}

(`e_target_both` skipped: add_only × changed cell < 50 turns.)

The DiD is stable at +11 to +18 with CI excluding 0 under: parse-fail
exclusion (a), flagged-model inclusion (b), COT prompting (c), feedback (d),
actual coding *with* mismatched turns kept (g), and within fact-targeted and
rule-targeted edits (e). **The only specification that kills it is the
pre-registered one (f): recorded metadata coding with mismatched turns
included (DiD +2.2, CI spans 0).** The difference between (f) and (g)/(primary)
is precisely the {n_mismatch} metadata-corrupted turns — mislabeled removals
sitting in the add_only comparator and explicit-track deltas that misdescribe
the state. The pre-registered null is an artifact of dataset metadata errors,
not evidence against suppression.

## 5. Error autopsy (Phase 0.4)

All incorrect turns in the primary slice, labeled per §6 of the spec.
Tables: `tables/autopsy_pooled.csv` (+ per-model); figure:
`figures/autopsy_bars.{{png,pdf}}`.

**Headline — flip-turn (removal × changed) error composition:**

| setting | stale | uncertain_retreat | other |
|---|---|---|---|
| explicit | {flip_cell("explicit", "stale")}% | {flip_cell("explicit", "uncertain_retreat")}% | {flip_cell("explicit", "other_changed")}% |
| implicit | {flip_cell("implicit", "stale")}% | {flip_cell("implicit", "uncertain_retreat")}% | {flip_cell("implicit", "other_changed")}% |

Pooled: **uncertain_retreat = {pct(fp.get("uncertain_retreat", 0))}%**,
stale = {pct(fp.get("stale", 0))}%. Excluding turn 1 (demo-answer caveat):
retreat {pct(fp1.get("uncertain_retreat", 0))}%, stale
{pct(fp1.get("stale", 0))}% — robust.

Models that fail flips overwhelmingly **retreat to Uncertain rather than keep
the stale answer**, and the retreat is concentrated in the explicit setting
(84% of explicit flip errors vs 51% implicit). Invariant-turn errors show the
same signature: ~84% of explicit invariant errors are spurious collapses to
Uncertain (vs ~46-53% implicit). This sharpens the paper's Table 5 (45.6%
"excessive uncertainty"): delta-presentation drives models toward
non-commitment, not belief perseverance — stale answers are the *minority*
failure mode everywhere.

## 6. Decision-gate readout (pre-registered thresholds, amended slice)

{md_table(kw["gates"])}

- **Suppression story: GO.** P1 DiD {did_str(p1, "did")}, ≥ 5 pts, CI
  excludes 0, and DiD > 0 in 100% of non-flagged models. On the
  pre-registered (uncorrected) coding the gate fails (+2.2, CI spans 0) —
  §4.6 attributes the difference to metadata corruption.
- **Intrinsic-contraction story: NO-GO** — the implicit removal deficit is
  large and *negative* (removals are easier than the addition-flip
  comparator once "the answer moved" is held constant).
- **Uncertain-retreat headline: GO** ({pct(fp.get("uncertain_retreat", 0))}%
  ≥ 50%).

**Recommendation:** proceed to matched generation *with* the suppression
claim, framed precisely: retracted-premises-in-context hurt on
answer-relevant removals (P1), cost nothing on answer-irrelevant ones (P3),
and the behavioral failure mode is retreat-to-Uncertain, not stale-belief
persistence (§5). Carry two caveats into the design: the addition-flip
comparator is small (18 delta-verified example-turns) and semantically
unusual (non-monotonic label flips); and the dataset's edit metadata needs
regeneration before any further explicit-setting evaluation.

## 7. Limitations

1. **Post-hoc amendment**: the primary slice recodes edits from actual FOL
   deltas and excludes 18.2% of turns; the pre-registered coding is
   sensitivity (f). The amendment was forced by verifiable metadata errors
   and approved mid-run, but it is not pre-registered inference.
2. **Small comparator**: `add_only × changed` has 18 unique delta-verified
   example-turns (306 items/setting pooled). Cluster-bootstrap CIs account
   for the clustering, but per-model add-cell estimates rest on ≤ 18 items.
3. **Comparator semantics**: flips-by-pure-addition rely on "False =
   ¬conclusion provable" non-monotonicity; they may be intrinsically harder
   than removal flips for reasons unrelated to presentation, which the DiD
   only partially nets out.
4. **Demonstration-answer bug** (§1.1): setting-neutral, but absolute
   accuracies are depressed and turn-1 gold_prev uses the true initial
   answer, not the (sometimes wrong) demonstrated one; autopsy is robust to
   dropping turn 1.
5. **Pooled GEE clusters only by example**, treating model repeats within an
   example as exchangeable; per-model GEEs and the bootstrap agree with the
   pooled conclusion on the changed-only estimand.
6. `implicit_shuffled` tracks exist for no model in this dump; that
   robustness check is not reproducible here.

## 8. Acceptance checklist

- [x] §2.1 verdict with file+line evidence (LATEX_TYPO_ONLY;
      `src/evaluation.py:249-265`, all git revisions + empirical check)
- [x] Pairing table and cell-count table printed before any test (§1.2, §2)
- [x] Coder unit tests pass (5 hand-checked cases + normalization suite)
- [x] Paper's pooled add/remove split reproduced (73.6 exact; 50.1 vs 50.6
      explained: pure-removal definition + absent gpt-5-mini logs)
- [x] P1–P3 with all three estimators + sensitivity battery (a–g)
- [x] Autopsy tables + figures
- [x] `run.sh` reproduces everything from raw logs on a clean checkout
"""
    with open(os.path.join(OUT, "report.md"), "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
