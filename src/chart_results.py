#!/usr/bin/env python3
import os
import json
import time
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_results(json_dir: str):
    """Read all JSON files in json_dir and return a list of parsed dicts."""
    results = []
    for fname in sorted(os.listdir(json_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(json_dir, fname)
        with open(path, "r") as f:
            results.append(json.load(f))
    return results


def make_degradation_plot(df: pd.DataFrame, out_dir: str, track: str):
    """Line plot of normalized accuracy vs. token count for one track,
    with x-axis ticks at each bucket boundary."""
    sns.set_style("whitegrid")
    plt.figure()
    sns.lineplot(
        data=df,
        x="tokens",
        y="accuracy",
        hue="model",
        marker="o",
        err_style=None,
    )
    ticks = sorted(df["tokens"].unique())
    plt.xticks(ticks, ticks)
    title = track.replace("_", " ").title()
    plt.title(f"{title} Degradation vs. Token Count (Normalized)")
    plt.xlabel("Tokens (bucket start)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()

    epoch = int(time.time())
    fname = f"{epoch}_{track}_degradation_norm.png"
    plt.savefig(os.path.join(out_dir, fname))
    plt.close()
    print(f"  • Saved degradation plot → {track}/{fname}")


def make_permutations_plot(df: pd.DataFrame, out_dir: str, track: str):
    """Bar chart of normalized accuracy for all permutation tags in one plot."""
    sns.set_style("whitegrid")
    plt.figure()
    sns.barplot(
        data=df,
        x="tag",
        y="accuracy",
        hue="model",
        errorbar=None,
    )
    title = track.replace("_", " ").title()
    plt.title(f"{title} Permutation Accuracy (Normalized)")
    plt.xlabel("Edit Tag")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    epoch = int(time.time())
    fname = f"{epoch}_{track}_permutations_norm.png"
    plt.savefig(os.path.join(out_dir, fname))
    plt.close()
    print(f"  • Saved combined permutation plot → {track}/{fname}")


def make_difficulty_plot(df: pd.DataFrame, out_dir: str, track: str):
    """Bar chart of normalized performance by difficulty."""
    # Compute normalized accuracy per model across difficulty levels
    df = df.copy()
    # Use transform instead of apply to keep index alignment
    df['accuracy'] = df.groupby('model')['count'].transform(lambda x: x / x.sum())

    sns.set_style("whitegrid")
    plt.figure()
    sns.barplot(
        data=df,
        x="difficulty",
        y="accuracy",
        hue="model",
        errorbar=None
    )
    title = track.replace("_", " ").title()
    plt.title(f"{title} Performance by Difficulty (Normalized)")
    plt.xlabel("Difficulty")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()

    epoch = int(time.time())
    fname = f"{epoch}_{track}_difficulty_norm.png"
    plt.savefig(os.path.join(out_dir, fname))
    plt.close()
    print(f"  • Saved difficulty plot → {track}/{fname}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate run JSONs and plot per-track charts"
    )
    parser.add_argument(
        "--json-dir",
        type=str,
        default="results",
        help="Directory containing run JSON files",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/charts",
        help="Directory in which to save chart PNGs",
    )
    args = parser.parse_args()

    runs = load_results(args.json_dir)

    degradation = {}     # track -> list of {model, tokens, accuracy}
    permutations = {}    # track -> list of {model, tag, accuracy}
    difficulty = {}      # track -> list of {model, difficulty, count}

    for run in runs:
        model = run.get("metadata", {}).get("model_name", "<unknown>")

        for track, buckets in run.get("degradation_buckets", {}).items():
            degradation.setdefault(track, [])
            merged = {"total": 0, "correct": 0}
            for b_str, stats in buckets.items():
                b = int(b_str)
                if b in (0, 1):
                    merged["total"]   += stats["total"]
                    merged["correct"] += stats["correct"]
            degradation[track].append({
                "model":    model,
                "tokens":   1024,
                "accuracy": merged["correct"] / merged["total"] if merged["total"] else 0.0
            })
            for b_str, stats in buckets.items():
                b = int(b_str)
                if b <= 1:
                    continue
                total, correct = stats["total"], stats["correct"]
                degradation[track].append({
                    "model":    model,
                    "tokens":   b * 512,
                    "accuracy": correct / total if total else 0.0
                })

        for track, tag_map in run.get("permutation_stats", {}).items():
            permutations.setdefault(track, [])
            for tag, stats in tag_map.items():
                total, correct = stats.get("total", 0), stats.get("correct", 0)
                accuracy = correct / total if total else 0.0
                permutations[track].append({
                    "model":    model,
                    "tag":      tag,
                    "accuracy": accuracy,
                })

        for track, counts in run.get("length_by_difficulty", {}).items():
            difficulty.setdefault(track, [])
            for lvl in ("easy", "medium", "hard"):
                difficulty[track].append({
                    "model":      model,
                    "difficulty": lvl.title(),
                    "count":      counts.get(lvl, 0),
                })

    for track in sorted(set(degradation) | set(permutations) | set(difficulty)):
        track_dir = os.path.join(args.out_dir, track)
        os.makedirs(track_dir, exist_ok=True)
        print(f"\nTrack: {track}")

        if track in degradation:
            df_deg = pd.DataFrame(degradation[track])
            make_degradation_plot(df_deg, track_dir, track)

        if track in permutations:
            df_perm = pd.DataFrame(permutations[track])
            make_permutations_plot(df_perm, track_dir, track)

        if track in difficulty:
            df_diff = pd.DataFrame(difficulty[track])
            make_difficulty_plot(df_diff, track_dir, track)


if __name__ == "__main__":
    main()
