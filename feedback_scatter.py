#!/usr/bin/env python3
"""
Create scatter plot: Feedback vs No Feedback accuracy by model
"""

import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("Loading data...")
    json_files = glob.glob("detailed_models_results/**/*.json", recursive=True)

    models_with_feedback = {}
    models_no_feedback = {}

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        model_name = data.get('metadata', {}).get('model_name', 'unknown')
        if model_name not in models_with_feedback:
            models_with_feedback[model_name] = {'correct': 0, 'total': 0}
        if model_name not in models_no_feedback:
            models_no_feedback[model_name] = {'correct': 0, 'total': 0}

        found_false = False
        for pred in data.get('predictions', []):
            if pred.get('is_demonstration', False) or pred.get('step', 0) == 0:
                continue

            if found_false:
                found_false = False
                # "no_correction" in path = WITHOUT correction/feedback
                # No "no_correction" in path = WITH correction/feedback
                if "no_correction" not in json_file:  # WITH correction
                    models_with_feedback[model_name]['total'] += 1
                    if pred.get("correct"):
                        models_with_feedback[model_name]['correct'] += 1
                else:  # WITHOUT correction
                    models_no_feedback[model_name]['total'] += 1
                    if pred.get("correct"):
                        models_no_feedback[model_name]['correct'] += 1

            if not pred.get("correct"):
                found_false = True

    # Calculate accuracies
    scatter_data = []
    for model in models_with_feedback.keys():
        if model == 'unknown':
            continue

        with_fb = models_with_feedback.get(model, {'correct': 0, 'total': 0})
        no_fb = models_no_feedback.get(model, {'correct': 0, 'total': 0})

        if with_fb['total'] > 0 and no_fb['total'] > 0:
            with_acc = with_fb['correct'] / with_fb['total']
            no_acc = no_fb['correct'] / no_fb['total']

            scatter_data.append({
                'model': model.split('/')[-1],
                'with_feedback': with_acc,
                'no_feedback': no_acc,
                'difference': with_acc - no_acc
            })

    df = pd.DataFrame(scatter_data)

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Store min/max for later use
    max_val = max(df['no_feedback'].max(), df['with_feedback'].max())
    min_val = min(df['no_feedback'].min(), df['with_feedback'].min())

    # Plot points - larger for visibility
    scatter = ax.scatter(df['no_feedback'], df['with_feedback'],
                        s=120, alpha=0.7, c='steelblue', edgecolors='black', linewidth=1.5)

    # Add labels for each point
    for idx, row in df.iterrows():
        # Clean up model names for readability
        model_name = row['model']
        # Remove common prefixes and simplify names
        clean_name = (model_name
                     .replace('qwen-', 'Qwen ')
                     .replace('qwen3-', 'Qwen 3 ')
                     .replace('gemma-3-', 'Gemma 3 ')
                     .replace('gpt-', 'GPT-')
                     .replace('claude-', 'Claude ')
                     .replace('gemini-2.5-', 'Gemini 2.5 ')
                     .replace('grok-code-', 'Grok ')
                     .replace('kimi-', 'Kimi '))

        ax.annotate(clean_name,
                   (row['no_feedback'], row['with_feedback']),
                   fontsize=12, alpha=0.9, fontweight='semibold',
                   xytext=(5, 5), textcoords='offset points')

    # Styling
    ax.set_xlabel('Without Feedback (Accuracy)', fontsize=16, fontweight='bold')
    ax.set_ylabel('With Feedback (Accuracy)', fontsize=16, fontweight='bold')
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
    vertices_fb = [(ax_min, ax_min), (ax_max, ax_max), (ax_min, ax_max)]
    poly_fb = Polygon(vertices_fb, alpha=0.15, facecolor='green', edgecolor='none', zorder=0)
    ax.add_patch(poly_fb)

    # Region where No Feedback is better (below diagonal)
    vertices_nofb = [(ax_min, ax_min), (ax_max, ax_min), (ax_max, ax_max)]
    poly_nofb = Polygon(vertices_nofb, alpha=0.15, facecolor='red', edgecolor='none', zorder=0)
    ax.add_patch(poly_nofb)

    # Add diagonal line
    line, = ax.plot([0, ax_max], [0, ax_max], 'k--', alpha=0.6, linewidth=2.5, label='Feedback = No Feedback', zorder=1)
    ax.legend(handles=[line], fontsize=13, loc='upper left', framealpha=0.95)

    # Calculate stats
    avg_diff = df['difference'].mean()
    below_diagonal = (df['difference'] < 0).sum()
    total = len(df)

    plt.tight_layout()
    plt.savefig('feedback_vs_no_feedback_scatter.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('feedback_vs_no_feedback_scatter.png', dpi=300, bbox_inches='tight')
    print(f"\nScatter plot saved to: feedback_vs_no_feedback_scatter.pdf/png")

    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Models favoring Feedback: {sum(df['difference'] > 0)}")
    print(f"Models favoring No Feedback: {sum(df['difference'] < 0)}")
    print(f"Models with similar performance: {sum(abs(df['difference']) < 0.01)}")
    print(f"\nAverage Feedback advantage: {avg_diff:.4f}")
    print(f"Median Feedback advantage: {df['difference'].median():.4f}")
    print(f"\nModels ranked by feedback impact:")
    print(df.sort_values('difference')[['model', 'no_feedback', 'with_feedback', 'difference']].to_string(index=False))

if __name__ == "__main__":
    main()
