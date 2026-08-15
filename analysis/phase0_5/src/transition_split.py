"""Phase 0.5 §1.2 — Transition split within removal × changed.

Splits the explicit→implicit gain and the error autopsy (stale vs
uncertain_retreat) by transition direction T→F vs F→T, on the Phase-0
primary slice. Prediction under the recommitment story: retreat-to-Uncertain
dominates both directions (rules out "F is just hard to say").

Writes tables/transition_split.csv and figures/transition_split.{png,pdf}.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(REPO, "analysis", "phase0_5")
sys.path.insert(0, os.path.join(REPO, "analysis", "phase0", "src"))
from analysis import make_slice, flagged_models, autopsy_label  # noqa: E402

C_ADD, C_REM, C_YEL, C_OTHER = "#2a78d6", "#eb6834", "#eda100", "#9a9992"


def main():
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    df = pd.read_parquet(os.path.join(REPO, "analysis", "phase0", "tidy.parquet"))
    prim = make_slice(df, flagged=flagged_models(df))
    sub = prim[(prim.edit_class == "removal") & prim.answer_changed
               & prim.transition.isin(["T→F", "F→T"])].copy()

    acc = (sub.groupby(["transition", "setting"], observed=True)
              .correct.agg(acc="mean", n="size").reset_index())
    acc["acc"] = (100 * acc.acc).round(2)
    wide = acc.pivot(index="transition", columns="setting", values="acc")
    wide["gain"] = (wide.implicit - wide.explicit).round(2)

    sub["autopsy"] = sub.apply(autopsy_label, axis=1)
    err = sub[sub.autopsy != "correct"]
    aut = (err.groupby(["transition", "setting", "autopsy"], observed=True)
              .size().rename("n").reset_index())
    tot = aut.groupby(["transition", "setting"], observed=True).n.transform("sum")
    aut["share"] = (100 * aut.n / tot).round(2)

    rows = []
    for tr in ["T→F", "F→T"]:
        row = {"transition": tr,
               "acc_explicit": wide.loc[tr, "explicit"],
               "acc_implicit": wide.loc[tr, "implicit"],
               "gain_impl_minus_expl": wide.loc[tr, "gain"]}
        for s in ["explicit", "implicit"]:
            for cat in ["stale", "uncertain_retreat", "other_changed"]:
                v = aut[(aut.transition == tr) & (aut.setting == s) & (aut.autopsy == cat)]
                row[f"{cat}_share_{s}"] = float(v.share.iloc[0]) if len(v) else 0.0
            row[f"n_errors_{s}"] = int(aut[(aut.transition == tr)
                                           & (aut.setting == s)].n.sum())
        row["n_turns_per_setting"] = int(acc[(acc.transition == tr)
                                             & (acc.setting == "explicit")].n.iloc[0])
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT, "tables", "transition_split.csv"), index=False)
    print(table.to_string(index=False))

    # figure: left = accuracy slopes by transition; right = retreat share bars
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax = axes[0]
    for tr, col in [("T→F", C_REM), ("F→T", C_ADD)]:
        y = [wide.loc[tr, "explicit"], wide.loc[tr, "implicit"]]
        ax.plot([0, 1], y, "-o", color=col, lw=2, ms=6)
        ax.annotate(tr, (1, y[1]), xytext=(6, 0), textcoords="offset points",
                    fontsize=9, va="center", color="#0b0b0b")
    ax.set_xticks([0, 1], ["explicit", "implicit"])
    ax.set_xlim(-0.2, 1.45)
    ax.set_ylim(0, 100)
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Removal flips: accuracy by transition", fontsize=10)

    ax = axes[1]
    groups = [(tr, s) for tr in ["T→F", "F→T"] for s in ["explicit", "implicit"]]
    xs = np.arange(len(groups))
    bottom = np.zeros(len(groups))
    for cat, col in [("stale", C_REM), ("uncertain_retreat", C_YEL),
                     ("other_changed", C_OTHER)]:
        vals = np.array([float(table.loc[table.transition == tr,
                                         f"{cat}_share_{s}"].iloc[0])
                         for tr, s in groups])
        ax.bar(xs, vals, bottom=bottom, color=col, width=0.62,
               edgecolor="#fcfcfb", linewidth=1.5, label=cat)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 8:
                ax.text(xi, b + v / 2, f"{v:.0f}", ha="center", va="center",
                        fontsize=7, color="#0b0b0b")
        bottom += vals
    ax.set_xticks(xs, [f"{tr}\n{s[:4]}" for tr, s in groups], fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("share of errors (%)")
    ax.set_title("Error composition by transition", fontsize=10)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.18))
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color="#e5e4df", lw=0.7)
        a.set_axisbelow(True)
    fig.suptitle("Phase 0.5 §1.2 — transition split (removal × changed, primary slice)",
                 fontsize=10)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(OUT, "figures", f"transition_split.{ext}"),
                    dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
