import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the COT vs Standard comparison data
cot_data = pd.read_csv('analysis_tables/table2_cot_vs_standard_overall.csv')

# Extract COT and Standard scores
models = cot_data['model'].values
cot_scores = cot_data['COT'].values
standard_scores = cot_data['Standard'].values

# Create the scatter plot
fig, ax = plt.subplots(figsize=(12, 10))

# Plot points - larger for visibility in 2-column format
scatter = ax.scatter(standard_scores, cot_scores, s=120, alpha=0.7, c='steelblue', edgecolors='black', linewidth=1.5)

# Store min/max for later use
max_val = max(max(cot_scores), max(standard_scores))
min_val = min(min(cot_scores), min(standard_scores))

# Add labels for each model
for i, model in enumerate(models):
    # Clean up model names for readability
    clean_name = model.replace('qwen-', '').replace('gemma-', '').replace('gpt-', '').replace('claude-', '')
    ax.annotate(clean_name, (standard_scores[i], cot_scores[i]),
                fontsize=12, alpha=0.9, fontweight='semibold',
                xytext=(5, 5), textcoords='offset points')

# Styling
ax.set_xlabel('Standard Prompting (Average LCAT)', fontsize=16, fontweight='bold')
ax.set_ylabel('Chain-of-Thought (Average LCAT)', fontsize=16, fontweight='bold')
ax.set_title('Model Performance: COT vs Standard Prompting', fontsize=18, fontweight='bold', pad=20)
ax.grid(True, alpha=0.2)

# Add clearer region labels positioned in their actual regions
ax.text(0.05, 0.95, 'COT Better\n(above line)', transform=ax.transAxes,
        fontsize=14, alpha=0.9, ha='left', va='top',
        fontweight='bold', color='darkgreen',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3, edgecolor='green', linewidth=2.5))
ax.text(0.95, 0.05, 'Standard Better\n(below line)', transform=ax.transAxes,
        fontsize=14, alpha=0.9, ha='right', va='bottom',
        fontweight='bold', color='darkred',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3, edgecolor='red', linewidth=2.5))

# Set equal aspect ratio
ax.set_aspect('equal', adjustable='box')

# Ensure both axes have the same range
ax_min = min(min_val, ax.get_xlim()[0], ax.get_ylim()[0]) - 0.02
ax_max = max(max_val, ax.get_xlim()[1], ax.get_ylim()[1]) + 0.02
ax.set_xlim(ax_min, ax_max)
ax.set_ylim(ax_min, ax_max)

# Add shaded regions AFTER setting axis limits
from matplotlib.patches import Polygon
# Region where COT is better (above diagonal)
vertices_cot = [(ax_min, ax_min), (ax_max, ax_max), (ax_min, ax_max)]
poly_cot = Polygon(vertices_cot, alpha=0.15, facecolor='green', edgecolor='none', zorder=0)
ax.add_patch(poly_cot)

# Region where Standard is better (below diagonal)
vertices_std = [(ax_min, ax_min), (ax_max, ax_min), (ax_max, ax_max)]
poly_std = Polygon(vertices_std, alpha=0.15, facecolor='red', edgecolor='none', zorder=0)
ax.add_patch(poly_std)

# Add diagonal line from 0,0 to max of figure
line, = ax.plot([0, ax_max], [0, ax_max], 'k--', alpha=0.6, linewidth=2.5, label='COT = Standard', zorder=1)
ax.legend(handles=[line], fontsize=13, loc='upper left', framealpha=0.95)

plt.tight_layout()
plt.savefig('figs/discussion/cot_vs_standard_scatter.pdf', bbox_inches='tight')
plt.savefig('figs/discussion/cot_vs_standard_scatter.png', dpi=300, bbox_inches='tight')
print(f"Scatter plot saved to figs/discussion/cot_vs_standard_scatter.pdf and .png")

# Print summary statistics
print("\nSummary Statistics:")
print(f"Models favoring COT: {sum(cot_scores > standard_scores)}")
print(f"Models favoring Standard: {sum(standard_scores > cot_scores)}")
print(f"Models with similar performance: {sum(abs(cot_scores - standard_scores) < 0.01)}")
print(f"\nAverage COT advantage: {np.mean(cot_scores - standard_scores):.4f}")
print(f"Median COT advantage: {np.median(cot_scores - standard_scores):.4f}")

# Don't show interactively
# plt.show()
