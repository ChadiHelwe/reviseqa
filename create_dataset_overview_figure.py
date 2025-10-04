import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Set style for publication
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'

# Create figure with custom layout
fig = plt.figure(figsize=(14, 8))

# Define layout
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4,
                      left=0.08, right=0.95, top=0.93, bottom=0.07)

# Color scheme
colors = {
    'primary': '#3498db',
    'secondary': '#2ecc71',
    'accent': '#e74c3c',
    'warning': '#f39c12',
    'light': '#ecf0f1',
    'dark': '#2c3e50'
}

# Title
fig.suptitle('ReviseQA Dataset Composition and Statistics',
             fontsize=16, fontweight='bold', y=0.97)

# ============================================================================
# 1. Dataset Size Overview (top left)
# ============================================================================
ax1 = fig.add_subplot(gs[0, 0])
sizes = [930, 6510, 19530]
labels = ['Examples', 'Edits', 'Verification\nTasks']
bars = ax1.bar(labels, sizes, color=[colors['primary'], colors['secondary'], colors['accent']])
ax1.set_ylabel('Count', fontweight='bold')
ax1.set_title('Dataset Scale', fontweight='bold', pad=10)
ax1.set_yscale('log')
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# ============================================================================
# 2. Answer Distribution (top center)
# ============================================================================
ax2 = fig.add_subplot(gs[0, 1])
answer_data = [453, 473, 4]
answer_labels = ['True\n(48.7%)', 'False\n(50.9%)', 'Uncertain\n(0.4%)']
wedges, texts = ax2.pie(answer_data, labels=answer_labels,
                        colors=[colors['secondary'], colors['accent'], colors['warning']],
                        startangle=90, textprops={'fontsize': 9, 'fontweight': 'bold'})
ax2.set_title('Answer Distribution', fontweight='bold', pad=10)

# ============================================================================
# 3. Modification Types (top right)
# ============================================================================
ax3 = fig.add_subplot(gs[0, 2])
mod_data = [3356, 3153, 1]
mod_labels = ['FLIP\n(51.6%)', 'INVARIANT\n(48.4%)', 'Uncertain\n(0.0%)']
wedges, texts = ax3.pie(mod_data, labels=mod_labels,
                        colors=[colors['accent'], colors['primary'], colors['warning']],
                        startangle=90, textprops={'fontsize': 9, 'fontweight': 'bold'})
ax3.set_title('Modification Types', fontweight='bold', pad=10)

# ============================================================================
# 4. Context Complexity (middle left)
# ============================================================================
ax4 = fig.add_subplot(gs[1, :])
metrics = ['Context\nStatements', 'Unique\nPredicates', 'Reasoning\nSteps']
means = [13.00, 13.36, 7.50]
stds = [2.18, 2.23, 1.14]

x = np.arange(len(metrics))
bars = ax4.bar(x, means, yerr=stds, capsize=5,
              color=[colors['primary'], colors['secondary'], colors['accent']],
              alpha=0.8, edgecolor='black', linewidth=1.5)

ax4.set_ylabel('Count', fontweight='bold')
ax4.set_title('Context Complexity (Mean ± SD)', fontweight='bold', pad=10)
ax4.set_xticks(x)
ax4.set_xticklabels(metrics, fontweight='bold')
ax4.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + std + 0.3,
            f'{mean:.2f}±{std:.2f}',
            ha='center', va='bottom', fontsize=8, fontweight='bold')

# ============================================================================
# 5. Model Performance (bottom left and center)
# ============================================================================
ax5 = fig.add_subplot(gs[2, :2])
models = ['Gemini-2.5-Flash', 'GPT-5-Mini', 'Qwen3-235B']
verified = [6465, 6438, 5381]
total = [6510, 6510, 6510]
accuracy = [99.3, 98.9, 82.7]

x = np.arange(len(models))
bars = ax5.bar(x, accuracy, color=[colors['secondary'], colors['primary'], colors['warning']],
              alpha=0.8, edgecolor='black', linewidth=1.5)

ax5.set_ylabel('Verification Accuracy (%)', fontweight='bold')
ax5.set_title('Model Verification Performance', fontweight='bold', pad=10)
ax5.set_xticks(x)
ax5.set_xticklabels(models, rotation=15, ha='right', fontweight='bold')
ax5.set_ylim(0, 105)
ax5.axhline(y=100, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax5.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for i, (bar, acc, ver, tot) in enumerate(zip(bars, accuracy, verified, total)):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{acc:.1f}%\n({ver:,}/{tot:,})',
            ha='center', va='bottom', fontsize=8, fontweight='bold')

# ============================================================================
# 6. Key Statistics Box (bottom right)
# ============================================================================
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')

# Create a fancy box for statistics
stats_text = """KEY STATISTICS

Operator Usage:
  → (implies): 5,120
  ⊕ (xor): 2,963
  ¬ (not): 2,819
  ∀ (forall): 2,096

Edit Operations:
  Avg Facts Added: 1.03
  Avg Rules Added: 1.11
  Avg Facts Removed: 0.35
  Avg Rules Removed: 0.63

Reasoning:
  Total Steps: 6,972
  With Conclusions: 95.1%
"""

# Add box
bbox = FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                      boxstyle="round,pad=0.05",
                      edgecolor=colors['dark'],
                      facecolor=colors['light'],
                      linewidth=2,
                      transform=ax6.transAxes)
ax6.add_patch(bbox)

ax6.text(0.5, 0.5, stats_text,
        transform=ax6.transAxes,
        ha='center', va='center',
        fontsize=8, fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.5))

# ============================================================================
# Save figure
# ============================================================================
plt.savefig('analysis_output/dataset_overview_comprehensive.pdf',
            bbox_inches='tight', dpi=300)
plt.savefig('analysis_output/dataset_overview_comprehensive.png',
            bbox_inches='tight', dpi=300)

print("✅ Comprehensive dataset overview figure saved!")
print("   - analysis_output/dataset_overview_comprehensive.pdf")
print("   - analysis_output/dataset_overview_comprehensive.png")

plt.show()
