#!/usr/bin/env python3
"""
Generate comprehensive CSV tables for all analyses.
Creates organized output for easy interpretation and paper writing.
"""

import pandas as pd
import numpy as np
import os

# Load data
df = pd.read_csv('combined_model_results.csv')

# Add classifications
df['method'] = df['task'].apply(lambda x: 'Standard' if 'no_reasoning' in x else 'COT')
df['has_correction'] = df['task'].apply(lambda x: 'No Correction' if 'no_correction' in x else 'With Correction')
df['context_type'] = df['task'].apply(lambda x: 'Implicit' if x.startswith('implicit') else 'Explicit')

# Create output directory
os.makedirs('analysis_tables', exist_ok=True)

print("Generating analysis tables...")
print("="*70)

# ============================================================================
# Table 1: Overall Model Performance Summary
# ============================================================================
print("\n1. Generating overall model performance summary...")

model_overall = df.groupby('model').agg({
    'score': ['mean', 'std', 'min', 'max', 'count']
}).round(4)
model_overall.columns = ['avg_score', 'std_score', 'min_score', 'max_score', 'n_observations']
model_overall = model_overall.sort_values('avg_score', ascending=False)
model_overall = model_overall.reset_index()

# Add rank
model_overall.insert(0, 'rank', range(1, len(model_overall) + 1))

model_overall.to_csv('analysis_tables/table1_overall_performance.csv', index=False)
print(f"   Saved: table1_overall_performance.csv ({len(model_overall)} models)")

# ============================================================================
# Table 2: COT vs Standard Performance (Overall)
# ============================================================================
print("\n2. Generating COT vs Standard comparison...")

model_method_avg = df.groupby(['model', 'method'])['score'].mean().reset_index()
cot_comparison = model_method_avg.pivot(index='model', columns='method', values='score')
cot_comparison['difference'] = cot_comparison['COT'] - cot_comparison['Standard']
cot_comparison['cot_advantage_pct'] = (cot_comparison['difference'] * 100).round(2)
cot_comparison['better_method'] = cot_comparison['difference'].apply(
    lambda x: 'COT' if x > 0.01 else ('Standard' if x < -0.01 else 'Similar')
)
cot_comparison = cot_comparison.round(4)
cot_comparison = cot_comparison.sort_values('difference', ascending=False)
cot_comparison = cot_comparison.reset_index()

cot_comparison.to_csv('analysis_tables/table2_cot_vs_standard_overall.csv', index=False)
print(f"   Saved: table2_cot_vs_standard_overall.csv")

# ============================================================================
# Table 3: COT vs Standard by K-value
# ============================================================================
print("\n3. Generating COT vs Standard by k-value...")

cot_by_k_list = []
for k_val in [7, 4, 2]:
    df_k = df[df['k'] == k_val]
    model_method_k = df_k.groupby(['model', 'method'])['score'].mean().reset_index()
    comparison_k = model_method_k.pivot(index='model', columns='method', values='score')
    comparison_k['difference'] = comparison_k['COT'] - comparison_k['Standard']
    comparison_k['k'] = k_val
    comparison_k = comparison_k.reset_index()
    cot_by_k_list.append(comparison_k)

cot_by_k = pd.concat(cot_by_k_list)
cot_by_k = cot_by_k[['model', 'k', 'Standard', 'COT', 'difference']]
cot_by_k = cot_by_k.sort_values(['k', 'difference'], ascending=[False, False])
cot_by_k = cot_by_k.round(4)

cot_by_k.to_csv('analysis_tables/table3_cot_vs_standard_by_k.csv', index=False)
print(f"   Saved: table3_cot_vs_standard_by_k.csv")

# ============================================================================
# Table 4: COT vs Standard by Task Type and K-value
# ============================================================================
print("\n4. Generating COT vs Standard by task type and k-value...")

task_pairs = {
    'explicit': ('explicit', 'explicit_no_reasoning'),
    'implicit': ('implicit', 'implicit_no_reasoning'),
    'explicit_no_corr': ('explicit_no_correction', 'explicit_no_reasoning_no_correction'),
    'implicit_no_corr': ('implicit_no_correction', 'implicit_no_reasoning_no_correction'),
}

cot_task_k_list = []
for k_val in [7, 4, 2]:
    df_k = df[df['k'] == k_val]

    for task_label, (cot_task, standard_task) in task_pairs.items():
        cot_data = df_k[df_k['task'] == cot_task][['model', 'score']].copy()
        standard_data = df_k[df_k['task'] == standard_task][['model', 'score']].copy()

        merged = cot_data.merge(standard_data, on='model', suffixes=('_COT', '_Standard'))
        merged['difference'] = merged['score_COT'] - merged['score_Standard']
        merged['k'] = k_val
        merged['task_type'] = task_label

        cot_task_k_list.append(merged)

cot_task_k = pd.concat(cot_task_k_list)
cot_task_k = cot_task_k[['model', 'k', 'task_type', 'score_Standard', 'score_COT', 'difference']]
cot_task_k = cot_task_k.sort_values(['k', 'task_type', 'difference'], ascending=[False, True, False])
cot_task_k = cot_task_k.round(4)

cot_task_k.to_csv('analysis_tables/table4_cot_by_task_and_k.csv', index=False)
print(f"   Saved: table4_cot_by_task_and_k.csv")

# ============================================================================
# Table 5: Explicit vs Implicit Performance
# ============================================================================
print("\n5. Generating Explicit vs Implicit comparison...")

model_context = df.groupby(['model', 'context_type'])['score'].mean().reset_index()
context_comparison = model_context.pivot(index='model', columns='context_type', values='score')
context_comparison['difference'] = context_comparison['Implicit'] - context_comparison['Explicit']
context_comparison['implicit_advantage_pct'] = (context_comparison['difference'] * 100).round(2)
context_comparison['stronger_on'] = context_comparison['difference'].apply(
    lambda x: 'Implicit' if x > 0.01 else ('Explicit' if x < -0.01 else 'Similar')
)
context_comparison = context_comparison.round(4)
context_comparison = context_comparison.sort_values('difference', ascending=False)
context_comparison = context_comparison.reset_index()

context_comparison.to_csv('analysis_tables/table5_explicit_vs_implicit.csv', index=False)
print(f"   Saved: table5_explicit_vs_implicit.csv")

# ============================================================================
# Table 6: Explicit vs Implicit by K-value
# ============================================================================
print("\n6. Generating Explicit vs Implicit by k-value...")

context_k_list = []
for k_val in [7, 4, 2]:
    df_k = df[df['k'] == k_val]
    model_context_k = df_k.groupby(['model', 'context_type'])['score'].mean().reset_index()
    comparison_k = model_context_k.pivot(index='model', columns='context_type', values='score')
    comparison_k['difference'] = comparison_k['Implicit'] - comparison_k['Explicit']
    comparison_k['k'] = k_val
    comparison_k = comparison_k.reset_index()
    context_k_list.append(comparison_k)

context_k = pd.concat(context_k_list)
context_k = context_k[['model', 'k', 'Explicit', 'Implicit', 'difference']]
context_k = context_k.sort_values(['k', 'difference'], ascending=[False, False])
context_k = context_k.round(4)

context_k.to_csv('analysis_tables/table6_explicit_vs_implicit_by_k.csv', index=False)
print(f"   Saved: table6_explicit_vs_implicit_by_k.csv")

# ============================================================================
# Table 7: Correction vs No Correction (Overall)
# ============================================================================
print("\n7. Generating Correction vs No Correction comparison...")

model_corr_avg = df.groupby(['model', 'has_correction'])['score'].mean().reset_index()
corr_comparison = model_corr_avg.pivot(index='model', columns='has_correction', values='score')
corr_comparison['difference'] = corr_comparison['With Correction'] - corr_comparison['No Correction']
corr_comparison['correction_benefit_pct'] = (corr_comparison['difference'] * 100).round(2)
corr_comparison['better_with'] = corr_comparison['difference'].apply(
    lambda x: 'Correction' if x > 0.01 else ('No Correction' if x < -0.01 else 'Similar')
)
corr_comparison = corr_comparison.round(4)
corr_comparison = corr_comparison.sort_values('difference', ascending=False)
corr_comparison = corr_comparison.reset_index()

corr_comparison.to_csv('analysis_tables/table7_correction_vs_no_correction.csv', index=False)
print(f"   Saved: table7_correction_vs_no_correction.csv")

# ============================================================================
# Table 8: Correction by K-value and Task Type
# ============================================================================
print("\n8. Generating Correction by task type and k-value...")

task_pairs_corr = {
    'explicit_cot': ('explicit', 'explicit_no_correction'),
    'explicit_standard': ('explicit_no_reasoning', 'explicit_no_reasoning_no_correction'),
    'implicit_cot': ('implicit', 'implicit_no_correction'),
    'implicit_standard': ('implicit_no_reasoning', 'implicit_no_reasoning_no_correction'),
}

corr_task_k_list = []
for k_val in [7, 4, 2]:
    df_k = df[df['k'] == k_val]

    for task_label, (with_corr_task, no_corr_task) in task_pairs_corr.items():
        with_corr_data = df_k[df_k['task'] == with_corr_task][['model', 'score']].copy()
        no_corr_data = df_k[df_k['task'] == no_corr_task][['model', 'score']].copy()

        merged = with_corr_data.merge(no_corr_data, on='model', suffixes=('_WithCorr', '_NoCorr'))
        merged['difference'] = merged['score_WithCorr'] - merged['score_NoCorr']
        merged['k'] = k_val
        merged['task_type'] = task_label

        corr_task_k_list.append(merged)

corr_task_k = pd.concat(corr_task_k_list)
corr_task_k = corr_task_k[['model', 'k', 'task_type', 'score_WithCorr', 'score_NoCorr', 'difference']]
corr_task_k = corr_task_k.sort_values(['k', 'task_type', 'difference'], ascending=[False, True, False])
corr_task_k = corr_task_k.round(4)

corr_task_k.to_csv('analysis_tables/table8_correction_by_task_and_k.csv', index=False)
print(f"   Saved: table8_correction_by_task_and_k.csv")

# ============================================================================
# Table 9: Performance by K-value (Overall)
# ============================================================================
print("\n9. Generating performance by k-value...")

model_k = df.groupby(['model', 'k'])['score'].mean().reset_index()
k_pivot = model_k.pivot(index='model', columns='k', values='score')
k_pivot.columns = [f'k_{int(k)}' for k in k_pivot.columns]
k_pivot['improvement_k7_to_k2'] = k_pivot['k_2'] - k_pivot['k_7']
k_pivot['improvement_pct'] = (k_pivot['improvement_k7_to_k2'] * 100).round(2)
k_pivot = k_pivot.round(4)
k_pivot = k_pivot.sort_values('improvement_k7_to_k2', ascending=False)
k_pivot = k_pivot.reset_index()

k_pivot.to_csv('analysis_tables/table9_performance_by_k.csv', index=False)
print(f"   Saved: table9_performance_by_k.csv")

# ============================================================================
# Table 10: Three-Way Interaction (Context × Method × Correction)
# ============================================================================
print("\n10. Generating three-way interaction table...")

three_way = df.groupby(['context_type', 'method', 'has_correction']).agg({
    'score': ['mean', 'std', 'count']
}).round(4)
three_way.columns = ['mean_score', 'std_score', 'n_models']
three_way = three_way.reset_index()
three_way = three_way.sort_values(['context_type', 'method', 'has_correction'])

three_way.to_csv('analysis_tables/table10_three_way_interaction.csv', index=False)
print(f"   Saved: table10_three_way_interaction.csv")

# ============================================================================
# Table 11: Task-Level Summary Statistics
# ============================================================================
print("\n11. Generating task-level summary statistics...")

task_summary = df.groupby('task').agg({
    'score': ['mean', 'std', 'min', 'max']
}).round(4)
task_summary.columns = ['mean_score', 'std_score', 'min_score', 'max_score']
task_summary = task_summary.reset_index()

# Add classifications
task_summary['method'] = task_summary['task'].apply(
    lambda x: 'Standard' if 'no_reasoning' in x else 'COT'
)
task_summary['correction'] = task_summary['task'].apply(
    lambda x: 'No Correction' if 'no_correction' in x else 'With Correction'
)
task_summary['context'] = task_summary['task'].apply(
    lambda x: 'Implicit' if x.startswith('implicit') else 'Explicit'
)

task_summary = task_summary[['task', 'context', 'method', 'correction',
                              'mean_score', 'std_score', 'min_score', 'max_score']]
task_summary = task_summary.sort_values('mean_score', ascending=False)

task_summary.to_csv('analysis_tables/table11_task_summary.csv', index=False)
print(f"   Saved: table11_task_summary.csv")

# ============================================================================
# Table 12: Model Size vs Performance
# ============================================================================
print("\n12. Generating model size vs performance...")

model_sizes = pd.read_csv('model_sizes.csv')
model_avg = df.groupby('model')['score'].mean().reset_index()
model_avg.columns = ['model', 'avg_score']

size_perf = model_sizes.merge(model_avg, on='model')
size_perf['size_numeric'] = size_perf['size_billions'].apply(
    lambda x: float(str(x).replace('~', '')) if pd.notna(x) else np.nan
)
size_perf['efficiency'] = (size_perf['avg_score'] / size_perf['size_numeric'] * 100).round(2)
size_perf = size_perf.sort_values('efficiency', ascending=False)

size_perf_out = size_perf[['model', 'size_billions', 'architecture', 'avg_score', 'efficiency']]
size_perf_out = size_perf_out.round(4)

size_perf_out.to_csv('analysis_tables/table12_model_size_performance.csv', index=False)
print(f"   Saved: table12_model_size_performance.csv")

# ============================================================================
# Table 13: Summary Statistics (For Paper)
# ============================================================================
print("\n13. Generating summary statistics table...")

summary_stats = []

# COT effect
cot_effect = df.groupby('method')['score'].mean()
summary_stats.append({
    'effect': 'COT vs Standard',
    'condition_1': 'COT',
    'score_1': cot_effect['COT'],
    'condition_2': 'Standard',
    'score_2': cot_effect['Standard'],
    'difference': cot_effect['COT'] - cot_effect['Standard'],
    'better': 'COT' if cot_effect['COT'] > cot_effect['Standard'] else 'Standard'
})

# Context effect
context_effect = df.groupby('context_type')['score'].mean()
summary_stats.append({
    'effect': 'Implicit vs Explicit',
    'condition_1': 'Implicit',
    'score_1': context_effect['Implicit'],
    'condition_2': 'Explicit',
    'score_2': context_effect['Explicit'],
    'difference': context_effect['Implicit'] - context_effect['Explicit'],
    'better': 'Implicit'
})

# Correction effect
corr_effect = df.groupby('has_correction')['score'].mean()
summary_stats.append({
    'effect': 'Correction vs No Correction',
    'condition_1': 'With Correction',
    'score_1': corr_effect['With Correction'],
    'condition_2': 'No Correction',
    'score_2': corr_effect['No Correction'],
    'difference': corr_effect['With Correction'] - corr_effect['No Correction'],
    'better': 'Correction' if corr_effect['With Correction'] > corr_effect['No Correction'] else 'No Correction'
})

# K-value effect
k_effect = df.groupby('k')['score'].mean().sort_index(ascending=False)
summary_stats.append({
    'effect': 'K-value (k=2 vs k=7)',
    'condition_1': 'k=2 (Easy)',
    'score_1': k_effect[2],
    'condition_2': 'k=7 (Hard)',
    'score_2': k_effect[7],
    'difference': k_effect[2] - k_effect[7],
    'better': 'k=2 (more corrections better)'
})

summary_df = pd.DataFrame(summary_stats)
summary_df = summary_df.round(4)

summary_df.to_csv('analysis_tables/table13_summary_statistics.csv', index=False)
print(f"   Saved: table13_summary_statistics.csv")

# ============================================================================
# Table 14: Top Performers by Category
# ============================================================================
print("\n14. Generating top performers by category...")

top_performers = []

# Overall top 5
overall_top = df.groupby('model')['score'].mean().nlargest(5)
for rank, (model, score) in enumerate(overall_top.items(), 1):
    top_performers.append({
        'category': 'Overall',
        'rank': rank,
        'model': model,
        'score': score
    })

# COT top 5
df_cot = df[df['method'] == 'COT']
cot_top = df_cot.groupby('model')['score'].mean().nlargest(5)
for rank, (model, score) in enumerate(cot_top.items(), 1):
    top_performers.append({
        'category': 'COT Reasoning',
        'rank': rank,
        'model': model,
        'score': score
    })

# Explicit reasoning top 5
df_explicit = df[df['context_type'] == 'Explicit']
explicit_top = df_explicit.groupby('model')['score'].mean().nlargest(5)
for rank, (model, score) in enumerate(explicit_top.items(), 1):
    top_performers.append({
        'category': 'Explicit Reasoning',
        'rank': rank,
        'model': model,
        'score': score
    })

# Implicit reasoning top 5
df_implicit = df[df['context_type'] == 'Implicit']
implicit_top = df_implicit.groupby('model')['score'].mean().nlargest(5)
for rank, (model, score) in enumerate(implicit_top.items(), 1):
    top_performers.append({
        'category': 'Implicit Reasoning',
        'rank': rank,
        'model': model,
        'score': score
    })

# Correction utilization top 5
model_corr_benefit = model_corr_avg.pivot(index='model', columns='has_correction', values='score')
model_corr_benefit['benefit'] = model_corr_benefit['With Correction'] - model_corr_benefit['No Correction']
corr_top = model_corr_benefit['benefit'].nlargest(5)
for rank, (model, benefit) in enumerate(corr_top.items(), 1):
    top_performers.append({
        'category': 'Correction Utilization',
        'rank': rank,
        'model': model,
        'score': benefit
    })

# Efficiency top 5
efficiency_top = size_perf_out.head(5)
for rank, (_, row) in enumerate(efficiency_top.iterrows(), 1):
    top_performers.append({
        'category': 'Efficiency (Score/Billion)',
        'rank': rank,
        'model': row['model'],
        'score': row['efficiency']
    })

top_performers_df = pd.DataFrame(top_performers)
top_performers_df = top_performers_df.round(4)

top_performers_df.to_csv('analysis_tables/table14_top_performers.csv', index=False)
print(f"   Saved: table14_top_performers.csv")

print("\n" + "="*70)
print("All analysis tables generated successfully!")
print("Output directory: analysis_tables/")
print(f"Total tables: 14")
print("="*70)
