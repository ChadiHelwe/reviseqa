#!/usr/bin/env python3
"""
Generate publication-quality figures for ACL paper discussion section.
Analyzes COT vs Standard, Explicit vs Implicit, and Correction effects.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Set publication-quality style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Load data
df = pd.read_csv('combined_model_results.csv')

# Add classifications
df['method'] = df['task'].apply(lambda x: 'Standard' if 'no_reasoning' in x else 'COT')
df['has_correction'] = df['task'].apply(lambda x: 'No Correction' if 'no_correction' in x else 'With Correction')
df['context_type'] = df['task'].apply(lambda x: 'Implicit' if x.startswith('implicit') else 'Explicit')

# Create output directory
import os
os.makedirs('figs/discussion', exist_ok=True)

print("Generating figures for ACL paper discussion section...")

# ============================================================================
# Figure 1: COT vs Standard Performance by K-value
# ============================================================================
print("\n1. Generating COT vs Standard comparison...")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
fig.suptitle('Chain-of-Thought vs. Standard Prompting Across Difficulty Levels', fontsize=13, y=1.02)

k_values = [7, 4, 2]
for idx, k_val in enumerate(k_values):
    ax = axes[idx]
    df_k = df[df['k'] == k_val]

    # Get average scores by model and method
    model_method = df_k.groupby(['model', 'method'])['score'].mean().reset_index()
    comparison = model_method.pivot(index='model', columns='method', values='score')
    comparison['difference'] = comparison['COT'] - comparison['Standard']
    comparison = comparison.sort_values('difference')

    # Create bar plot
    x = np.arange(len(comparison))
    width = 0.35

    bars1 = ax.barh(x - width/2, comparison['Standard'], width, label='Standard',
                     color='#E8927C', alpha=0.8)
    bars2 = ax.barh(x + width/2, comparison['COT'], width, label='COT',
                     color='#69B3E7', alpha=0.8)

    ax.set_yticks(x)
    ax.set_yticklabels([m.replace('-', '\n', 1) if len(m) > 15 else m
                        for m in comparison.index], fontsize=7)
    ax.set_xlabel('Accuracy', fontsize=10)
    ax.set_title(f'k={k_val} ({"Hard" if k_val==7 else "Medium" if k_val==4 else "Easy"})',
                 fontsize=11)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(0, 1.0)

    # Add statistics
    cot_better = (comparison['difference'] > 0.01).sum()
    cot_pct = cot_better / len(comparison) * 100
    avg_diff = comparison['difference'].mean()
    ax.text(0.02, 0.98, f'COT better: {cot_pct:.0f}%\nΔ = +{avg_diff:.3f}',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('figs/discussion/fig1_cot_vs_standard_by_k.pdf', bbox_inches='tight')
plt.savefig('figs/discussion/fig1_cot_vs_standard_by_k.png', bbox_inches='tight')
print("   Saved: fig1_cot_vs_standard_by_k.pdf/png")

# ============================================================================
# Figure 2: COT Benefit by Task Type and K-value (Heatmap)
# ============================================================================
print("\n2. Generating COT benefit heatmap...")

task_pairs = {
    'Explicit': ('explicit', 'explicit_no_reasoning'),
    'Implicit': ('implicit', 'implicit_no_reasoning'),
    'Explicit\n(no corr)': ('explicit_no_correction', 'explicit_no_reasoning_no_correction'),
    'Implicit\n(no corr)': ('implicit_no_correction', 'implicit_no_reasoning_no_correction'),
}

heatmap_data = []
for k_val in [7, 4, 2]:
    row_data = []
    df_k = df[df['k'] == k_val]

    for task_label, (cot_task, standard_task) in task_pairs.items():
        cot_scores = df_k[df_k['task'] == cot_task]['score']
        standard_scores = df_k[df_k['task'] == standard_task]['score']

        diff = cot_scores.mean() - standard_scores.mean()
        row_data.append(diff)

    heatmap_data.append(row_data)

fig, ax = plt.subplots(figsize=(8, 3))
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
            xticklabels=list(task_pairs.keys()),
            yticklabels=['k=7 (Hard)', 'k=4 (Medium)', 'k=2 (Easy)'],
            cbar_kws={'label': 'COT Advantage'},
            vmin=-0.01, vmax=0.07, ax=ax)
ax.set_title('Chain-of-Thought Performance Advantage by Task Type and Difficulty', fontsize=12)
ax.set_xlabel('Task Type', fontsize=11)
ax.set_ylabel('Difficulty Level', fontsize=11)

plt.tight_layout()
plt.savefig('figs/discussion/fig2_cot_benefit_heatmap.pdf', bbox_inches='tight')
plt.savefig('figs/discussion/fig2_cot_benefit_heatmap.png', bbox_inches='tight')
print("   Saved: fig2_cot_benefit_heatmap.pdf/png")

# ============================================================================
# Figure 3: Explicit vs Implicit Performance Gap
# ============================================================================
print("\n3. Generating Explicit vs Implicit comparison...")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: Performance by context type and k-value
ax = axes[0]
context_k_scores = df.groupby(['context_type', 'k'])['score'].mean().reset_index()

x = np.array([0, 1, 2])
width = 0.35

explicit_scores = context_k_scores[context_k_scores['context_type'] == 'Explicit']['score'].values
implicit_scores = context_k_scores[context_k_scores['context_type'] == 'Implicit']['score'].values

bars1 = ax.bar(x - width/2, explicit_scores, width, label='Explicit',
               color='#E8927C', alpha=0.8)
bars2 = ax.bar(x + width/2, implicit_scores, width, label='Implicit',
               color='#69B3E7', alpha=0.8)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Difficulty Level (k value)', fontsize=11)
ax.set_ylabel('Average Accuracy', fontsize=11)
ax.set_title('Explicit vs. Implicit Context Performance', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(['k=7 (Hard)', 'k=4 (Medium)', 'k=2 (Easy)'])
ax.legend()
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 1.0)

# Right: Gap size decreases with easier tasks
ax = axes[1]
gaps = implicit_scores - explicit_scores
ax.plot(x, gaps, marker='o', linewidth=2, markersize=8, color='#E76F51')
ax.fill_between(x, 0, gaps, alpha=0.3, color='#E76F51')

for i, gap in enumerate(gaps):
    ax.text(x[i], gap + 0.01, f'{gap:.3f}', ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Difficulty Level (k value)', fontsize=11)
ax.set_ylabel('Performance Gap\n(Implicit - Explicit)', fontsize=11)
ax.set_title('Explicit-Implicit Gap Narrows with Easier Tasks', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(['k=7 (Hard)', 'k=4 (Medium)', 'k=2 (Easy)'])
ax.grid(alpha=0.3, linestyle='--')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig('figs/discussion/fig3_explicit_vs_implicit.pdf', bbox_inches='tight')
plt.savefig('figs/discussion/fig3_explicit_vs_implicit.png', bbox_inches='tight')
print("   Saved: fig3_explicit_vs_implicit.pdf/png")

# ============================================================================
# Figure 4: Top Model Performance on Explicit vs Implicit
# ============================================================================
print("\n4. Generating top model performance scatter...")

fig, ax = plt.subplots(figsize=(8, 7))

# Get average scores by model and context type
model_context = df.groupby(['model', 'context_type'])['score'].mean().reset_index()
model_pivot = model_context.pivot(index='model', columns='context_type', values='score')

# Get model sizes
model_sizes = pd.read_csv('model_sizes.csv')
model_pivot = model_pivot.merge(model_sizes[['model', 'size_billions']],
                                 left_index=True, right_on='model')

# Create scatter plot with size-based coloring
scatter = ax.scatter(model_pivot['Explicit'], model_pivot['Implicit'],
                     s=100, alpha=0.6, c=range(len(model_pivot)),
                     cmap='viridis', edgecolors='black', linewidth=0.5)

# Add model labels
for idx, row in model_pivot.iterrows():
    model_name = row['model'].replace('-', '\n', 1) if len(row['model']) > 15 else row['model']
    ax.annotate(model_name, (row['Explicit'], row['Implicit']),
                fontsize=7, ha='right', va='bottom',
                xytext=(-3, 3), textcoords='offset points')

# Add diagonal line (equal performance)
lims = [0, 1.0]
ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1, label='Equal Performance')

# Shading for implicit/explicit advantage
ax.fill_between(lims, lims, 1.0, alpha=0.1, color='blue', label='Implicit Advantage')
ax.fill_between(lims, 0, lims, alpha=0.1, color='red', label='Explicit Advantage')

ax.set_xlabel('Explicit Task Accuracy', fontsize=11)
ax.set_ylabel('Implicit Task Accuracy', fontsize=11)
ax.set_title('Model Performance: Explicit vs. Implicit Reasoning', fontsize=12)
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.0)
ax.legend(loc='lower right')
ax.grid(alpha=0.3, linestyle='--')
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('figs/discussion/fig4_model_explicit_vs_implicit_scatter.pdf', bbox_inches='tight')
plt.savefig('figs/discussion/fig4_model_explicit_vs_implicit_scatter.png', bbox_inches='tight')
print("   Saved: fig4_model_explicit_vs_implicit_scatter.pdf/png")

# ============================================================================
# Figure 5: Correction Impact by Task Type and K-value
# ============================================================================
print("\n5. Generating correction impact analysis...")

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Impact of Correction Feedback by Task Type and Difficulty', fontsize=13, y=0.995)

task_pairs_corr = {
    'Explicit\n(COT)': ('explicit', 'explicit_no_correction'),
    'Explicit\n(Standard)': ('explicit_no_reasoning', 'explicit_no_reasoning_no_correction'),
    'Implicit\n(COT)': ('implicit', 'implicit_no_correction'),
    'Implicit\n(Standard)': ('implicit_no_reasoning', 'implicit_no_reasoning_no_correction'),
}

for idx, k_val in enumerate([7, 4, 2]):
    if idx >= 3:
        break

    row = idx // 2
    col = idx % 2
    ax = axes[row, col]

    df_k = df[df['k'] == k_val]

    differences = []
    task_labels = []

    for task_label, (with_corr, no_corr) in task_pairs_corr.items():
        with_corr_scores = df_k[df_k['task'] == with_corr]['score']
        no_corr_scores = df_k[df_k['task'] == no_corr]['score']

        diff = with_corr_scores.mean() - no_corr_scores.mean()
        differences.append(diff)
        task_labels.append(task_label)

    colors = ['#E8927C' if diff > 0 else '#69B3E7' for diff in differences]
    bars = ax.barh(task_labels, differences, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    # Add value labels
    for i, (bar, diff) in enumerate(zip(bars, differences)):
        x_pos = diff + (0.002 if diff > 0 else -0.002)
        ha = 'left' if diff > 0 else 'right'
        ax.text(x_pos, i, f'{diff:+.3f}', ha=ha, va='center', fontsize=8)

    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Correction Benefit\n(With Corr - No Corr)', fontsize=10)
    ax.set_title(f'k={k_val} ({"Hard" if k_val==7 else "Medium" if k_val==4 else "Easy"})', fontsize=11)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(-0.02, 0.02)

# Summary plot (bottom right)
ax = axes[1, 1]
k_values = [7, 4, 2]
avg_benefits = []

for k_val in k_values:
    df_k = df[df['k'] == k_val]
    with_corr = df_k[df_k['has_correction'] == 'With Correction']['score'].mean()
    no_corr = df_k[df_k['has_correction'] == 'No Correction']['score'].mean()
    avg_benefits.append(with_corr - no_corr)

ax.plot(k_values, avg_benefits, marker='o', linewidth=2, markersize=10, color='#E76F51')
ax.fill_between(k_values, 0, avg_benefits, alpha=0.3, color='#E76F51')

for k, benefit in zip(k_values, avg_benefits):
    ax.text(k, benefit + 0.001, f'{benefit:+.3f}', ha='center', va='bottom', fontsize=9)

ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Difficulty Level (k value)', fontsize=10)
ax.set_ylabel('Average Correction Benefit', fontsize=10)
ax.set_title('Overall Correction Impact Trend', fontsize=11)
ax.set_xticks(k_values)
ax.set_xticklabels(['k=7\n(Hard)', 'k=4\n(Medium)', 'k=2\n(Easy)'])
ax.grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('figs/discussion/fig5_correction_impact.pdf', bbox_inches='tight')
plt.savefig('figs/discussion/fig5_correction_impact.png', bbox_inches='tight')
print("   Saved: fig5_correction_impact.pdf/png")

# ============================================================================
# Figure 6: Model-Specific Correction Utilization
# ============================================================================
print("\n6. Generating model correction utilization...")

fig, ax = plt.subplots(figsize=(10, 6))

# Get correction benefit by model
model_corr = df.groupby(['model', 'has_correction'])['score'].mean().reset_index()
model_corr_pivot = model_corr.pivot(index='model', columns='has_correction', values='score')
model_corr_pivot['benefit'] = model_corr_pivot['With Correction'] - model_corr_pivot['No Correction']
model_corr_pivot = model_corr_pivot.sort_values('benefit', ascending=True)

colors = ['#69B3E7' if x < 0 else '#E8927C' for x in model_corr_pivot['benefit']]
bars = ax.barh(range(len(model_corr_pivot)), model_corr_pivot['benefit'],
               color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

ax.set_yticks(range(len(model_corr_pivot)))
ax.set_yticklabels([m.replace('-', '\n', 1) if len(m) > 15 else m
                     for m in model_corr_pivot.index], fontsize=8)
ax.set_xlabel('Correction Benefit (With Correction - No Correction)', fontsize=11)
ax.set_title('Model-Specific Correction Feedback Utilization', fontsize=12)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels
for i, (bar, benefit) in enumerate(zip(bars, model_corr_pivot['benefit'])):
    x_pos = benefit + (0.002 if benefit > 0 else -0.002)
    ha = 'left' if benefit > 0 else 'right'
    ax.text(x_pos, i, f'{benefit:+.3f}', ha=ha, va='center', fontsize=7)

# Highlight top performer
top_idx = len(model_corr_pivot) - 1
ax.axhspan(top_idx - 0.4, top_idx + 0.4, alpha=0.2, color='gold')
ax.text(0.98, 0.98, f'Top: {model_corr_pivot.index[-1]}\n{model_corr_pivot["benefit"].iloc[-1]:+.3f}',
        transform=ax.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='gold', alpha=0.3))

plt.tight_layout()
plt.savefig('figs/discussion/fig6_model_correction_utilization.pdf', bbox_inches='tight')
plt.savefig('figs/discussion/fig6_model_correction_utilization.png', bbox_inches='tight')
print("   Saved: fig6_model_correction_utilization.pdf/png")

# ============================================================================
# Figure 7: Combined Analysis - COT, Context Type, and Correction
# ============================================================================
print("\n7. Generating combined three-way analysis...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Comprehensive Analysis: COT, Context Type, and Correction Effects', fontsize=14, y=0.995)

# Top left: COT benefit by context type
ax = axes[0, 0]
context_method = df.groupby(['context_type', 'method'])['score'].mean().reset_index()
context_pivot = context_method.pivot(index='context_type', columns='method', values='score')

x = np.arange(len(context_pivot))
width = 0.35
bars1 = ax.bar(x - width/2, context_pivot['Standard'], width, label='Standard',
               color='#E8927C', alpha=0.8)
bars2 = ax.bar(x + width/2, context_pivot['COT'], width, label='COT',
               color='#69B3E7', alpha=0.8)

ax.set_ylabel('Average Accuracy', fontsize=10)
ax.set_title('COT Benefit by Context Type', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(context_pivot.index)
ax.legend()
ax.grid(axis='y', alpha=0.3, linestyle='--')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# Top right: Correction benefit by context type
ax = axes[0, 1]
context_corr = df.groupby(['context_type', 'has_correction'])['score'].mean().reset_index()
context_corr_pivot = context_corr.pivot(index='context_type', columns='has_correction', values='score')

bars1 = ax.bar(x - width/2, context_corr_pivot['No Correction'], width,
               label='No Correction', color='#E8927C', alpha=0.8)
bars2 = ax.bar(x + width/2, context_corr_pivot['With Correction'], width,
               label='With Correction', color='#69B3E7', alpha=0.8)

ax.set_ylabel('Average Accuracy', fontsize=10)
ax.set_title('Correction Benefit by Context Type', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(context_corr_pivot.index)
ax.legend()
ax.grid(axis='y', alpha=0.3, linestyle='--')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# Bottom left: Three-way interaction heatmap
ax = axes[1, 0]
three_way = df.groupby(['context_type', 'method', 'has_correction'])['score'].mean().reset_index()

# Create matrix
matrix_data = []
row_labels = []
for context in ['Explicit', 'Implicit']:
    for method in ['Standard', 'COT']:
        row_label = f'{context}\n{method}'
        row_labels.append(row_label)

        no_corr = three_way[(three_way['context_type'] == context) &
                            (three_way['method'] == method) &
                            (three_way['has_correction'] == 'No Correction')]['score'].values[0]
        with_corr = three_way[(three_way['context_type'] == context) &
                              (three_way['method'] == method) &
                              (three_way['has_correction'] == 'With Correction')]['score'].values[0]
        matrix_data.append([no_corr, with_corr])

sns.heatmap(matrix_data, annot=True, fmt='.3f', cmap='YlOrRd',
            xticklabels=['No Correction', 'With Correction'],
            yticklabels=row_labels, cbar_kws={'label': 'Accuracy'}, ax=ax)
ax.set_title('Three-Way Interaction: Context × Method × Correction', fontsize=11)

# Bottom right: Summary statistics table
ax = axes[1, 1]
ax.axis('off')

# Calculate key statistics
stats_text = "Key Findings Summary\n" + "="*50 + "\n\n"

# COT effect
cot_effect = df.groupby('method')['score'].mean()
stats_text += f"1. Chain-of-Thought Effect:\n"
stats_text += f"   • COT: {cot_effect['COT']:.3f}\n"
stats_text += f"   • Standard: {cot_effect['Standard']:.3f}\n"
stats_text += f"   • Δ = +{cot_effect['COT'] - cot_effect['Standard']:.3f} (COT better)\n\n"

# Context effect
context_effect = df.groupby('context_type')['score'].mean()
stats_text += f"2. Context Type Effect:\n"
stats_text += f"   • Implicit: {context_effect['Implicit']:.3f}\n"
stats_text += f"   • Explicit: {context_effect['Explicit']:.3f}\n"
stats_text += f"   • Δ = +{context_effect['Implicit'] - context_effect['Explicit']:.3f} (Implicit easier)\n\n"

# Correction effect
corr_effect = df.groupby('has_correction')['score'].mean()
stats_text += f"3. Correction Feedback Effect:\n"
stats_text += f"   • With Correction: {corr_effect['With Correction']:.3f}\n"
stats_text += f"   • No Correction: {corr_effect['No Correction']:.3f}\n"
stats_text += f"   • Δ = +{corr_effect['With Correction'] - corr_effect['No Correction']:.3f}\n\n"

# Effect sizes
stats_text += "4. Relative Effect Sizes:\n"
cot_size = abs(cot_effect['COT'] - cot_effect['Standard'])
context_size = abs(context_effect['Implicit'] - context_effect['Explicit'])
corr_size = abs(corr_effect['With Correction'] - corr_effect['No Correction'])

stats_text += f"   • Context Type: {context_size:.3f} (largest)\n"
stats_text += f"   • COT: {cot_size:.3f} ({cot_size/corr_size:.1f}× correction)\n"
stats_text += f"   • Correction: {corr_size:.3f} (smallest)\n"

ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
        fontsize=9, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('figs/discussion/fig7_combined_analysis.pdf', bbox_inches='tight')
plt.savefig('figs/discussion/fig7_combined_analysis.png', bbox_inches='tight')
print("   Saved: fig7_combined_analysis.pdf/png")

# ============================================================================
# Figure 8: Model Size vs Performance Analysis
# ============================================================================
print("\n8. Generating model size analysis...")

# Load model sizes
model_sizes = pd.read_csv('model_sizes.csv')
model_avg_scores = df.groupby('model')['score'].mean().reset_index()
model_avg_scores.columns = ['model', 'avg_score']

size_performance = model_sizes.merge(model_avg_scores, on='model')

# Convert size to numeric, handling estimates
size_performance['size_numeric'] = size_performance['size_billions'].apply(
    lambda x: float(str(x).replace('~', '')) if pd.notna(x) else np.nan
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Scatter plot of size vs performance
ax = axes[0]
colors_arch = {'Standard': '#E8927C', 'MoE': '#69B3E7'}

for arch_type in ['Standard', 'MoE']:
    subset = size_performance[size_performance['architecture'] == arch_type]
    ax.scatter(subset['size_numeric'], subset['avg_score'],
               s=120, alpha=0.7, label=arch_type,
               color=colors_arch[arch_type], edgecolors='black', linewidth=0.5)

    # Add labels for notable models
    for _, row in subset.iterrows():
        if row['avg_score'] > 0.6 or row['size_numeric'] < 10 or row['size_numeric'] > 200:
            ax.annotate(row['model'].split('-')[0][:8],
                       (row['size_numeric'], row['avg_score']),
                       fontsize=7, ha='center', va='bottom',
                       xytext=(0, 3), textcoords='offset points')

ax.set_xlabel('Model Size (Billions of Parameters)', fontsize=11)
ax.set_ylabel('Average Accuracy', fontsize=11)
ax.set_title('Model Size vs. Performance', fontsize=12)
ax.set_xscale('log')
ax.legend()
ax.grid(alpha=0.3, linestyle='--')

# Add trendline for Standard models
standard_models = size_performance[size_performance['architecture'] == 'Standard'].dropna()
if len(standard_models) > 2:
    z = np.polyfit(np.log(standard_models['size_numeric']),
                   standard_models['avg_score'], 1)
    p = np.poly1d(z)
    x_trend = np.logspace(np.log10(standard_models['size_numeric'].min()),
                          np.log10(standard_models['size_numeric'].max()), 100)
    ax.plot(x_trend, p(np.log(x_trend)), "--", color='gray', alpha=0.5,
            linewidth=2, label='Standard trend')

# Right: Performance per parameter efficiency
ax = axes[1]
size_performance['efficiency'] = size_performance['avg_score'] / size_performance['size_numeric'] * 100

efficiency_sorted = size_performance.nlargest(10, 'efficiency')

colors = [colors_arch[arch] for arch in efficiency_sorted['architecture']]
bars = ax.barh(range(len(efficiency_sorted)), efficiency_sorted['efficiency'],
               color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

ax.set_yticks(range(len(efficiency_sorted)))
ax.set_yticklabels([f"{m[:20]}\n({s}B)"
                     for m, s in zip(efficiency_sorted['model'],
                                    efficiency_sorted['size_billions'])],
                    fontsize=8)
ax.set_xlabel('Efficiency Score\n(Accuracy / Billion Parameters × 100)', fontsize=10)
ax.set_title('Most Efficient Models (Performance per Parameter)', fontsize=11)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels
for i, (bar, eff, score) in enumerate(zip(bars, efficiency_sorted['efficiency'],
                                           efficiency_sorted['avg_score'])):
    ax.text(eff + 0.5, i, f'{eff:.1f}\n({score:.3f})',
            ha='left', va='center', fontsize=7)

plt.tight_layout()
plt.savefig('figs/discussion/fig8_model_size_analysis.pdf', bbox_inches='tight')
plt.savefig('figs/discussion/fig8_model_size_analysis.png', bbox_inches='tight')
print("   Saved: fig8_model_size_analysis.pdf/png")

print("\n" + "="*70)
print("All figures generated successfully!")
print("Output directory: figs/discussion/")
print("="*70)
