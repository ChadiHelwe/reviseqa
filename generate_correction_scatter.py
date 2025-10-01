import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the Correction vs No Correction comparison data
corr_data = pd.read_csv('analysis_tables/table7_correction_vs_no_correction.csv')

# Extract Correction and No Correction scores
models = corr_data['model'].values
with_correction = corr_data['With Correction'].values
no_correction = corr_data['No Correction'].values

# Create the scatter plot
fig, ax = plt.subplots(figsize=(12, 10))

# Plot points - larger for visibility in 2-column format
scatter = ax.scatter(no_correction, with_correction, s=120, alpha=0.7, c='steelblue', edgecolors='black', linewidth=1.5)

# Store min/max for later use
max_val = max(max(with_correction), max(no_correction))
min_val = min(min(with_correction), min(no_correction))

# Add labels for each model
for i, model in enumerate(models):
    # Clean up model names for readability
    clean_name = model.replace('qwen-', '').replace('gemma-', '').replace('gpt-', '').replace('claude-', '')
    ax.annotate(clean_name, (no_correction[i], with_correction[i]),
                fontsize=12, alpha=0.9, fontweight='semibold',
                xytext=(5, 5), textcoords='offset points')

# Styling
ax.set_xlabel('No Feedback (Average LCAT)', fontsize=16, fontweight='bold')
ax.set_ylabel('Feedback (Average LCAT)', fontsize=16, fontweight='bold')
ax.set_title('Model Performance: Feedback vs No Feedback', fontsize=18, fontweight='bold', pad=20)
ax.grid(True, alpha=0.2)

# Add clearer region labels positioned in their actual regions
ax.text(0.05, 0.95, 'Feedback Better\n(above line)', transform=ax.transAxes,
        fontsize=14, alpha=0.9, ha='left', va='top',
        fontweight='bold', color='darkgreen',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3, edgecolor='green', linewidth=2.5))
ax.text(0.95, 0.05, 'No Feedback Better\n(below line)', transform=ax.transAxes,
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
# Region where Feedback is better (above diagonal)
vertices_feedback = [(ax_min, ax_min), (ax_max, ax_max), (ax_min, ax_max)]
poly_feedback = Polygon(vertices_feedback, alpha=0.15, facecolor='green', edgecolor='none', zorder=0)
ax.add_patch(poly_feedback)

# Region where No Feedback is better (below diagonal)
vertices_no_feedback = [(ax_min, ax_min), (ax_max, ax_min), (ax_max, ax_max)]
poly_no_feedback = Polygon(vertices_no_feedback, alpha=0.15, facecolor='red', edgecolor='none', zorder=0)
ax.add_patch(poly_no_feedback)

# Add diagonal line from 0,0 to max of figure
line, = ax.plot([0, ax_max], [0, ax_max], 'k--', alpha=0.6, linewidth=2.5, label='Feedback = No Feedback', zorder=1)
ax.legend(handles=[line], fontsize=13, loc='upper left', framealpha=0.95)

plt.tight_layout()
plt.savefig('figs/discussion/correction_vs_no_correction_scatter.pdf', bbox_inches='tight')
plt.savefig('figs/discussion/correction_vs_no_correction_scatter.png', dpi=300, bbox_inches='tight')
print(f"Scatter plot saved to figs/discussion/correction_vs_no_correction_scatter.pdf and .png")

# Print summary statistics
print("\nSummary Statistics:")
print(f"Models favoring Feedback: {sum(with_correction > no_correction)}")
print(f"Models favoring No Feedback: {sum(no_correction > with_correction)}")
print(f"Models with similar performance: {sum(abs(with_correction - no_correction) < 0.01)}")
print(f"\nAverage Feedback advantage: {np.mean(with_correction - no_correction):.4f}")
print(f"Median Feedback advantage: {np.median(with_correction - no_correction):.4f}")

# Don't show interactively
# plt.show()
