#!/usr/bin/env python3
"""
Analyze whether models perform better with correction feedback or without.
"""

import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('combined_model_results.csv')

# Classify tasks by correction
df['has_correction'] = df['task'].apply(lambda x: 'No Correction' if 'no_correction' in x else 'With Correction')

# Also classify by reasoning for paired comparison
df['has_reasoning'] = df['task'].apply(lambda x: 'No Reasoning' if 'no_reasoning' in x else 'With Reasoning')

print("="*100)
print("CORRECTION vs NO CORRECTION Performance Analysis")
print("="*100)
print("\nWith Correction tasks: explicit, implicit, explicit_no_reasoning, implicit_no_reasoning")
print("No Correction tasks: explicit_no_correction, implicit_no_correction, etc.")
print("\n" + "="*100)

# Overall comparison
model_corr_avg = df.groupby(['model', 'has_correction'])['score'].mean().reset_index()
comparison = model_corr_avg.pivot(index='model', columns='has_correction', values='score')
comparison['difference'] = comparison['With Correction'] - comparison['No Correction']
comparison['better_method'] = comparison['difference'].apply(
    lambda x: 'Correction' if x > 0.01 else ('No Correction' if x < -0.01 else 'Similar')
)

comparison_sorted = comparison.sort_values('difference', ascending=False)

print(f"\n{'Model':<40} {'With Corr':>10} {'No Corr':>10} {'Diff':>10} {'Better'}")
print("-"*80)
for model, row in comparison_sorted.iterrows():
    print(f"{model:<40} {row['With Correction']:>10.3f} {row['No Correction']:>10.3f} {row['difference']:>10.3f} {row['better_method']}")

# Summary
corr_better = (comparison['difference'] > 0.01).sum()
no_corr_better = (comparison['difference'] < -0.01).sum()
similar = (abs(comparison['difference']) <= 0.01).sum()

print("\n" + "="*100)
print("SUMMARY STATISTICS")
print("="*100)
print(f"\nModels where Correction performs better: {corr_better} ({corr_better/len(comparison)*100:.1f}%)")
print(f"Models where No Correction performs better: {no_corr_better} ({no_corr_better/len(comparison)*100:.1f}%)")
print(f"Models with similar performance: {similar} ({similar/len(comparison)*100:.1f}%)")
print(f"\nAverage With Correction score: {comparison['With Correction'].mean():.3f}")
print(f"Average No Correction score: {comparison['No Correction'].mean():.3f}")
print(f"Overall difference (With - No): {comparison['difference'].mean():.3f}")

# Analysis by k-value and task type
print("\n" + "="*100)
print("ANALYSIS BY K-VALUE AND TASK TYPE")
print("="*100)

# Define task pairs for comparison
task_pairs = {
    'Explicit (with reasoning)': ('explicit', 'explicit_no_correction'),
    'Explicit (no reasoning)': ('explicit_no_reasoning', 'explicit_no_reasoning_no_correction'),
    'Implicit (with reasoning)': ('implicit', 'implicit_no_correction'),
    'Implicit (no reasoning)': ('implicit_no_reasoning', 'implicit_no_reasoning_no_correction'),
}

for k_val in [7, 4, 2]:
    print(f"\n{'='*100}")
    print(f"K = {k_val} (Maximum {k_val} correction steps)")
    print(f"{'='*100}")

    df_k = df[df['k'] == k_val]

    for task_label, (with_corr_task, no_corr_task) in task_pairs.items():
        print(f"\n{task_label}:")
        print("-"*100)

        # Get scores for this task pair at this k value
        with_corr_data = df_k[df_k['task'] == with_corr_task][['model', 'score']].copy()
        no_corr_data = df_k[df_k['task'] == no_corr_task][['model', 'score']].copy()

        # Merge to compare
        merged = with_corr_data.merge(no_corr_data, on='model', suffixes=('_WithCorr', '_NoCorr'))
        merged['difference'] = merged['score_WithCorr'] - merged['score_NoCorr']
        merged = merged.sort_values('difference', ascending=False)

        # Summary stats
        corr_better = (merged['difference'] > 0.01).sum()
        no_corr_better = (merged['difference'] < -0.01).sum()
        similar = (abs(merged['difference']) <= 0.01).sum()

        print(f"Models where With Correction better: {corr_better} ({corr_better/len(merged)*100:.0f}%), "
              f"No Correction better: {no_corr_better} ({no_corr_better/len(merged)*100:.0f}%), "
              f"Similar: {similar} ({similar/len(merged)*100:.0f}%)")
        print(f"Average - With Correction: {merged['score_WithCorr'].mean():.3f}, No Correction: {merged['score_NoCorr'].mean():.3f}, "
              f"Diff: {merged['difference'].mean():.3f}")

        print(f"\n{'Model':<40} {'With Corr':>10} {'No Corr':>10} {'Diff':>10}")
        print("-"*80)
        for _, row in merged.head(10).iterrows():
            print(f"{row['model']:<40} {row['score_WithCorr']:>10.3f} {row['score_NoCorr']:>10.3f} {row['difference']:>10.3f}")

print("\n" + "="*100)
print("OVERALL TASK COMPARISON")
print("="*100)

task_avg = df.groupby(['task'])['score'].mean().reset_index()

print(f"\n{'With Correction Task':<35} {'No Correction Task':<35} {'W/Corr':>10} {'No Corr':>10} {'Diff':>10}")
print("-"*100)
for task_label, (with_corr_task, no_corr_task) in task_pairs.items():
    with_score = task_avg[task_avg['task'] == with_corr_task]['score'].values
    no_score = task_avg[task_avg['task'] == no_corr_task]['score'].values

    if len(with_score) > 0 and len(no_score) > 0:
        diff = with_score[0] - no_score[0]
        print(f"{with_corr_task:<35} {no_corr_task:<35} {with_score[0]:>10.3f} {no_score[0]:>10.3f} {diff:>10.3f}")