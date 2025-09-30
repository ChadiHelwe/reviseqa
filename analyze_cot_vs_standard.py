#!/usr/bin/env python3
"""
Analyze whether models perform better with COT (Chain of Thought) or Standard methods.
COT tasks: Tasks WITHOUT "no_reasoning" (they include reasoning)
Standard tasks: Tasks WITH "no_reasoning" (they don't include reasoning)
"""

import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('combined_model_results.csv')

# Classify tasks into COT and Standard
# Tasks with "no_reasoning" = Standard (no COT)
# Tasks without "no_reasoning" = COT (with reasoning)
df['method'] = df['task'].apply(lambda x: 'Standard' if 'no_reasoning' in x else 'COT')

# Group by model and method to get average scores
model_method_avg = df.groupby(['model', 'method'])['score'].mean().reset_index()

# Pivot to compare COT vs Standard side by side
comparison = model_method_avg.pivot(index='model', columns='method', values='score')
comparison['difference'] = comparison['Standard'] - comparison['COT']
comparison['better_method'] = comparison['difference'].apply(
    lambda x: 'Standard' if x > 0.01 else ('COT' if x < -0.01 else 'Similar')
)

# Sort by difference
comparison_sorted = comparison.sort_values('difference', ascending=False)

print("="*80)
print("COT (Chain of Thought) vs Standard Method Performance Analysis")
print("="*80)
print("\nCOT tasks: Tasks WITHOUT 'no_reasoning' (they INCLUDE reasoning)")
print("  - explicit, implicit, explicit_no_correction, implicit_no_correction")
print("\nStandard tasks: Tasks WITH 'no_reasoning' (they DON'T include reasoning)")
print("  - explicit_no_reasoning, implicit_no_reasoning, etc.")
print("\nPositive difference = Standard performs better")
print("Negative difference = COT performs better")
print("\n" + "="*80)

print(f"\n{'Model':<40} {'Standard':>10} {'COT':>10} {'Diff':>10} {'Better'}")
print("-"*80)
for model, row in comparison_sorted.iterrows():
    print(f"{model:<40} {row['Standard']:>10.3f} {row['COT']:>10.3f} {row['difference']:>10.3f} {row['better_method']}")

# Overall statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

standard_better = (comparison['difference'] > 0.01).sum()
cot_better = (comparison['difference'] < -0.01).sum()
similar = (abs(comparison['difference']) <= 0.01).sum()

print(f"\nModels where Standard performs better: {standard_better} ({standard_better/len(comparison)*100:.1f}%)")
print(f"Models where COT performs better: {cot_better} ({cot_better/len(comparison)*100:.1f}%)")
print(f"Models with similar performance: {similar} ({similar/len(comparison)*100:.1f}%)")

print(f"\nAverage Standard score: {comparison['Standard'].mean():.3f}")
print(f"Average COT score: {comparison['COT'].mean():.3f}")
print(f"Overall difference (Standard - COT): {comparison['difference'].mean():.3f}")

# Task-level analysis
print("\n" + "="*80)
print("TASK-LEVEL ANALYSIS")
print("="*80)

task_method_avg = df.groupby(['task', 'method'])['score'].mean().reset_index()
print("\nAverage scores by task type:")
print(f"\n{'Task':<45} {'Method':<10} {'Avg Score':>10}")
print("-"*70)
for _, row in task_method_avg.sort_values(['task', 'method']).iterrows():
    print(f"{row['task']:<45} {row['method']:<10} {row['score']:>10.3f}")

# Analysis by k-value and task
print("\n" + "="*80)
print("ANALYSIS BY K-VALUE AND TASK TYPE")
print("="*80)

# Define task pairs for comparison
task_pairs = {
    'Explicit': ('explicit', 'explicit_no_reasoning'),
    'Implicit': ('implicit', 'implicit_no_reasoning'),
    'Explicit (no correction)': ('explicit_no_correction', 'explicit_no_reasoning_no_correction'),
    'Implicit (no correction)': ('implicit_no_correction', 'implicit_no_reasoning_no_correction'),
}

for k_val in [7, 4, 2]:
    print(f"\n{'='*100}")
    print(f"K = {k_val} (Maximum {k_val} correction steps)")
    print(f"{'='*100}")

    df_k = df[df['k'] == k_val]

    for task_label, (cot_task, standard_task) in task_pairs.items():
        print(f"\n{task_label}:")
        print("-"*100)

        # Get scores for this task pair at this k value
        cot_data = df_k[df_k['task'] == cot_task][['model', 'score']].copy()
        standard_data = df_k[df_k['task'] == standard_task][['model', 'score']].copy()

        # Merge to compare
        merged = cot_data.merge(standard_data, on='model', suffixes=('_COT', '_Standard'))
        merged['difference'] = merged['score_COT'] - merged['score_Standard']
        merged = merged.sort_values('difference', ascending=False)

        # Summary stats
        cot_better = (merged['difference'] > 0.01).sum()
        standard_better = (merged['difference'] < -0.01).sum()
        similar = (abs(merged['difference']) <= 0.01).sum()

        print(f"Models where COT better: {cot_better} ({cot_better/len(merged)*100:.0f}%), "
              f"Standard better: {standard_better} ({standard_better/len(merged)*100:.0f}%), "
              f"Similar: {similar} ({similar/len(merged)*100:.0f}%)")
        print(f"Average - COT: {merged['score_COT'].mean():.3f}, Standard: {merged['score_Standard'].mean():.3f}, "
              f"Diff: {merged['difference'].mean():.3f}")

        print(f"\n{'Model':<40} {'Standard':>10} {'COT':>10} {'Diff':>10}")
        print("-"*70)
        for _, row in merged.head(10).iterrows():
            print(f"{row['model']:<40} {row['score_Standard']:>10.3f} {row['score_COT']:>10.3f} {row['difference']:>10.3f}")

# Paired task comparison (explicit vs explicit_no_reasoning, etc.)
print("\n" + "="*80)
print("PAIRED TASK COMPARISON (With vs Without Reasoning)")
print("="*80)

# Get unique base tasks
task_avg = df.groupby(['task'])['score'].mean().reset_index()

pairs = [
    ('explicit', 'explicit_no_reasoning'),
    ('implicit', 'implicit_no_reasoning'),
    ('explicit_no_correction', 'explicit_no_reasoning_no_correction'),
    ('implicit_no_correction', 'implicit_no_reasoning_no_correction'),
]

print(f"\n{'COT Task':<35} {'Standard Task':<35} {'COT Score':>10} {'Std Score':>10} {'Diff':>10}")
print("-"*100)
for cot_task, standard_task in pairs:
    cot_score = task_avg[task_avg['task'] == cot_task]['score'].values
    standard_score = task_avg[task_avg['task'] == standard_task]['score'].values

    if len(cot_score) > 0 and len(standard_score) > 0:
        diff = cot_score[0] - standard_score[0]
        better = "COT" if diff > 0 else "Standard"
        print(f"{cot_task:<35} {standard_task:<35} {cot_score[0]:>10.3f} {standard_score[0]:>10.3f} {diff:>10.3f}")