#!/usr/bin/env python3
"""
Comprehensive analysis of accuracy differences between adding vs removing operations
in the ReviseQA detailed model results.
"""

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_and_parse_data(results_dir):
    """Load all JSON files and extract relevant data for analysis."""
    print("Loading and parsing data...")
    data = []
    files_processed = 0

    # Find all JSON files
    json_files = glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True)
    print(f"Found {len(json_files)} JSON files to process")

    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                json_data = json.load(f)

            # Extract metadata
            metadata = json_data.get('metadata', {})
            model_name = metadata.get('model_name', 'unknown')
            task_path = metadata.get('task_path', 'unknown')

            # Process each prediction step
            predictions = json_data.get('predictions', [])
            for pred in predictions:
                step = pred.get('step', 0)
                tags = pred.get('tags', [])
                correct = pred.get('correct', None)
                is_demonstration = pred.get('is_demonstration', False)

                # Skip demonstration steps (step 0)
                if is_demonstration or step == 0:
                    continue

                # Categorize operation type based on tags
                operation_type = categorize_operation(tags)

                if operation_type and correct is not None:
                    data.append({
                        'model_name': model_name,
                        'task_type': task_path,
                        'step': step,
                        'tags': tags,
                        'operation_type': operation_type,
                        'correct': correct,
                        'file_path': file_path
                    })

            files_processed += 1
            if files_processed % 1000 == 0:
                print(f"Processed {files_processed} files...")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    print(f"Successfully processed {files_processed} files")
    print(f"Extracted {len(data)} prediction steps for analysis")

    return pd.DataFrame(data)

def categorize_operation(tags):
    """Categorize operation type based on tags."""
    tags_set = set(tags)

    # Check for adding operations
    has_adding = bool(tags_set & {'added_rules', 'added_facts'})

    # Check for removing operations
    has_removing = bool(tags_set & {'removed_rules', 'removed_facts'})

    # Check for original
    has_original = 'original' in tags_set

    if has_adding and has_removing:
        return 'Mixed'
    elif has_adding:
        return 'Adding'
    elif has_removing:
        return 'Removing'
    elif has_original:
        return 'Original'
    else:
        return 'Other'

def calculate_accuracy_stats(df):
    """Calculate accuracy statistics by model, task, and operation type."""
    print("Calculating accuracy statistics...")

    # Group by model, task, and operation type
    grouped = df.groupby(['model_name', 'task_type', 'operation_type'])

    stats = []
    for (model, task, op_type), group in grouped:
        total_count = len(group)
        correct_count = group['correct'].sum()
        accuracy = correct_count / total_count if total_count > 0 else 0

        stats.append({
            'model_name': model,
            'task_type': task,
            'operation_type': op_type,
            'total_count': total_count,
            'correct_count': correct_count,
            'accuracy': accuracy
        })

    stats_df = pd.DataFrame(stats)

    # Filter to only include models/tasks with sufficient data for both Adding and Removing
    min_samples = 10  # Minimum samples required for reliable statistics

    filtered_stats = []
    for model in stats_df['model_name'].unique():
        for task in stats_df['task_type'].unique():
            model_task_data = stats_df[(stats_df['model_name'] == model) &
                                     (stats_df['task_type'] == task)]

            adding_data = model_task_data[model_task_data['operation_type'] == 'Adding']
            removing_data = model_task_data[model_task_data['operation_type'] == 'Removing']

            # Check if we have sufficient data for both operations
            adding_count = adding_data['total_count'].sum() if not adding_data.empty else 0
            removing_count = removing_data['total_count'].sum() if not removing_data.empty else 0

            if adding_count >= min_samples and removing_count >= min_samples:
                filtered_stats.extend(model_task_data.to_dict('records'))

    filtered_df = pd.DataFrame(filtered_stats)

    print(f"Found {len(filtered_df)} valid model-task-operation combinations")
    print(f"Models with sufficient data: {filtered_df['model_name'].nunique()}")
    print(f"Task types with sufficient data: {filtered_df['task_type'].nunique()}")

    return filtered_df

def create_bar_charts_by_task(stats_df, output_dir):
    """Create bar charts for each task type showing Adding vs Removing accuracy."""
    print("Creating bar charts by task type...")

    task_types = stats_df['task_type'].unique()

    for task_type in task_types:
        task_data = stats_df[stats_df['task_type'] == task_type]

        # Pivot data for easier plotting
        pivot_data = task_data.pivot_table(
            index='model_name',
            columns='operation_type',
            values='accuracy',
            fill_value=0
        )

        # Only plot if we have both Adding and Removing data
        if 'Adding' in pivot_data.columns and 'Removing' in pivot_data.columns:
            fig, ax = plt.subplots(figsize=(14, 8))

            # Create grouped bar chart
            x = np.arange(len(pivot_data.index))
            width = 0.35

            adding_accuracies = pivot_data['Adding'].values
            removing_accuracies = pivot_data['Removing'].values

            bars1 = ax.bar(x - width/2, adding_accuracies, width,
                          label='Adding Operations', alpha=0.8, color='#1f77b4')
            bars2 = ax.bar(x + width/2, removing_accuracies, width,
                          label='Removing Operations', alpha=0.8, color='#ff7f0e')

            # Customize plot
            ax.set_xlabel('Model')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'Accuracy Comparison: Adding vs Removing Operations\nTask Type: {task_type}',
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([name.split('/')[-1] for name in pivot_data.index],
                              rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)

            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)

            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)

            plt.tight_layout()

            # Save plot
            safe_task_name = task_type.replace('/', '_').replace(' ', '_')
            plt.savefig(os.path.join(output_dir, f'accuracy_bars_{safe_task_name}.pdf'),
                       dpi=300, bbox_inches='tight')
            plt.close()

def create_accuracy_difference_heatmap(stats_df, output_dir):
    """Create heatmap showing accuracy differences (Adding - Removing)."""
    print("Creating accuracy difference heatmap...")

    # Calculate differences
    pivot_adding = stats_df[stats_df['operation_type'] == 'Adding'].pivot_table(
        index='model_name', columns='task_type', values='accuracy', fill_value=np.nan)

    pivot_removing = stats_df[stats_df['operation_type'] == 'Removing'].pivot_table(
        index='model_name', columns='task_type', values='accuracy', fill_value=np.nan)

    # Calculate difference (Adding - Removing)
    diff_matrix = pivot_adding - pivot_removing

    # Remove rows/columns with all NaN
    diff_matrix = diff_matrix.dropna(how='all', axis=0).dropna(how='all', axis=1)

    if not diff_matrix.empty:
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create heatmap
        sns.heatmap(diff_matrix, annot=True, cmap='RdBu_r', center=0,
                   fmt='.3f', ax=ax, cbar_kws={'label': 'Accuracy Difference'})

        ax.set_title('Accuracy Difference: Adding vs Removing Operations\n(Adding - Removing)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Task Type')
        ax.set_ylabel('Model')

        # Improve readability
        ax.set_yticklabels([name.split('/')[-1] for name in diff_matrix.index], rotation=0)
        ax.set_xticklabels(diff_matrix.columns, rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_difference_heatmap.pdf'),
                   dpi=300, bbox_inches='tight')
        plt.close()

def create_scatter_plot(stats_df, output_dir):
    """Create scatter plot: Adding accuracy vs Removing accuracy by model."""
    print("Creating scatter plot...")

    # Prepare data for scatter plot
    scatter_data = []

    for model in stats_df['model_name'].unique():
        for task in stats_df['task_type'].unique():
            model_task_data = stats_df[(stats_df['model_name'] == model) &
                                     (stats_df['task_type'] == task)]

            adding_acc = model_task_data[model_task_data['operation_type'] == 'Adding']['accuracy'].values
            removing_acc = model_task_data[model_task_data['operation_type'] == 'Removing']['accuracy'].values

            if len(adding_acc) > 0 and len(removing_acc) > 0:
                scatter_data.append({
                    'model_name': model,
                    'task_type': task,
                    'adding_accuracy': adding_acc[0],
                    'removing_accuracy': removing_acc[0]
                })

    scatter_df = pd.DataFrame(scatter_data)

    if not scatter_df.empty:
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create scatter plot with different colors for task types
        task_types = scatter_df['task_type'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(task_types)))

        for i, task_type in enumerate(task_types):
            task_data = scatter_df[scatter_df['task_type'] == task_type]
            ax.scatter(task_data['adding_accuracy'], task_data['removing_accuracy'],
                      c=[colors[i]], label=task_type, alpha=0.7, s=60)

        # Add diagonal line (perfect correlation)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Correlation')

        # Customize plot
        ax.set_xlabel('Adding Operations Accuracy')
        ax.set_ylabel('Removing Operations Accuracy')
        ax.set_title('Adding vs Removing Accuracy by Model and Task Type',
                    fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'adding_vs_removing_scatter.pdf'),
                   dpi=300, bbox_inches='tight')
        plt.close()

def create_task_difference_line_plot(stats_df, output_dir):
    """Create line plot showing average differences across models for each task type."""
    print("Creating task difference line plot...")

    # Calculate average differences by task type
    task_differences = []

    for task_type in stats_df['task_type'].unique():
        task_data = stats_df[stats_df['task_type'] == task_type]

        model_diffs = []
        for model in task_data['model_name'].unique():
            model_task_data = task_data[task_data['model_name'] == model]

            adding_acc = model_task_data[model_task_data['operation_type'] == 'Adding']['accuracy'].values
            removing_acc = model_task_data[model_task_data['operation_type'] == 'Removing']['accuracy'].values

            if len(adding_acc) > 0 and len(removing_acc) > 0:
                diff = adding_acc[0] - removing_acc[0]
                model_diffs.append(diff)

        if model_diffs:
            task_differences.append({
                'task_type': task_type,
                'mean_difference': np.mean(model_diffs),
                'std_difference': np.std(model_diffs),
                'n_models': len(model_diffs)
            })

    task_diff_df = pd.DataFrame(task_differences)

    if not task_diff_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(task_diff_df))
        means = task_diff_df['mean_difference']
        stds = task_diff_df['std_difference']

        # Create line plot with error bars
        ax.errorbar(x, means, yerr=stds, marker='o', linewidth=2, markersize=8,
                   capsize=5, capthick=2)

        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # Customize plot
        ax.set_xlabel('Task Type')
        ax.set_ylabel('Mean Accuracy Difference\n(Adding - Removing)')
        ax.set_title('Average Accuracy Differences Across Models by Task Type',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(task_diff_df['task_type'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

        # Add value labels
        for i, (mean, std, n) in enumerate(zip(means, stds, task_diff_df['n_models'])):
            ax.text(i, mean + std + 0.01, f'{mean:.3f}\n(n={n})',
                   ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'task_difference_trends.pdf'),
                   dpi=300, bbox_inches='tight')
        plt.close()

def perform_statistical_analysis(df, stats_df, output_dir):
    """Perform statistical analysis and identify models with largest gaps."""
    print("Performing statistical analysis...")

    # Calculate differences for each model-task combination
    analysis_results = []

    for model in stats_df['model_name'].unique():
        model_data = stats_df[stats_df['model_name'] == model]

        model_diffs = []
        task_analyses = []

        for task in model_data['task_type'].unique():
            task_data = model_data[model_data['task_type'] == task]

            adding_data = task_data[task_data['operation_type'] == 'Adding']
            removing_data = task_data[task_data['operation_type'] == 'Removing']

            if not adding_data.empty and not removing_data.empty:
                adding_acc = adding_data['accuracy'].iloc[0]
                removing_acc = removing_data['accuracy'].iloc[0]
                diff = adding_acc - removing_acc

                model_diffs.append(diff)
                task_analyses.append({
                    'task_type': task,
                    'adding_accuracy': adding_acc,
                    'removing_accuracy': removing_acc,
                    'difference': diff,
                    'adding_count': adding_data['total_count'].iloc[0],
                    'removing_count': removing_data['total_count'].iloc[0]
                })

        if model_diffs:
            analysis_results.append({
                'model_name': model,
                'mean_difference': np.mean(model_diffs),
                'std_difference': np.std(model_diffs),
                'max_difference': np.max(model_diffs),
                'min_difference': np.min(model_diffs),
                'n_task_types': len(model_diffs),
                'task_analyses': task_analyses
            })

    # Sort by mean difference to identify models with largest gaps
    analysis_results.sort(key=lambda x: abs(x['mean_difference']), reverse=True)

    # Create summary report
    report_lines = [
        "=" * 80,
        "STATISTICAL ANALYSIS: ADDING VS REMOVING OPERATIONS",
        "=" * 80,
        "",
        f"Total models analyzed: {len(analysis_results)}",
        f"Total data points processed: {len(df)}",
        "",
        "TOP 10 MODELS WITH LARGEST ACCURACY GAPS:",
        "-" * 50
    ]

    for i, result in enumerate(analysis_results[:10]):
        model_short = result['model_name'].split('/')[-1]
        report_lines.extend([
            f"{i+1:2d}. {model_short}",
            f"    Mean difference: {result['mean_difference']:+.3f}",
            f"    Std deviation:   {result['std_difference']:.3f}",
            f"    Range: [{result['min_difference']:+.3f}, {result['max_difference']:+.3f}]",
            f"    Task types: {result['n_task_types']}",
            ""
        ])

    # Overall statistics
    all_diffs = [r['mean_difference'] for r in analysis_results]
    report_lines.extend([
        "OVERALL STATISTICS:",
        "-" * 30,
        f"Mean accuracy difference: {np.mean(all_diffs):+.3f}",
        f"Median accuracy difference: {np.median(all_diffs):+.3f}",
        f"Standard deviation: {np.std(all_diffs):.3f}",
        f"Range: [{np.min(all_diffs):+.3f}, {np.max(all_diffs):+.3f}]",
        "",
        "INTERPRETATION:",
        "-" * 20,
        "Positive values: Adding operations perform better than Removing",
        "Negative values: Removing operations perform better than Adding",
        "Values near zero: Similar performance between operations"
    ])

    # Save analysis report
    with open(os.path.join(output_dir, 'statistical_analysis_report.txt'), 'w') as f:
        f.write('\n'.join(report_lines))

    return analysis_results

def create_summary_csv(df, stats_df, analysis_results, output_dir):
    """Create a comprehensive summary CSV with all statistics."""
    print("Creating summary CSV...")

    # Detailed statistics by model-task-operation
    detailed_stats = []
    for _, row in stats_df.iterrows():
        detailed_stats.append({
            'model_name': row['model_name'],
            'model_short': row['model_name'].split('/')[-1],
            'task_type': row['task_type'],
            'operation_type': row['operation_type'],
            'accuracy': row['accuracy'],
            'total_count': row['total_count'],
            'correct_count': row['correct_count']
        })

    detailed_df = pd.DataFrame(detailed_stats)
    detailed_df.to_csv(os.path.join(output_dir, 'detailed_accuracy_stats.csv'), index=False)

    # Summary statistics by model
    summary_stats = []
    for result in analysis_results:
        summary_stats.append({
            'model_name': result['model_name'],
            'model_short': result['model_name'].split('/')[-1],
            'mean_difference': result['mean_difference'],
            'std_difference': result['std_difference'],
            'max_difference': result['max_difference'],
            'min_difference': result['min_difference'],
            'n_task_types': result['n_task_types']
        })

    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(os.path.join(output_dir, 'model_summary_stats.csv'), index=False)

    print(f"Saved detailed stats: {len(detailed_df)} rows")
    print(f"Saved summary stats: {len(summary_df)} rows")

def main():
    """Main analysis pipeline."""
    # Set up paths
    results_dir = "/Users/helwec/Desktop/reviseqa/detailed_models_results"
    output_dir = "/Users/helwec/Desktop/reviseqa/analysis_output"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("Starting comprehensive analysis of Adding vs Removing operations...")
    print(f"Input directory: {results_dir}")
    print(f"Output directory: {output_dir}")

    # Load and parse data
    df = load_and_parse_data(results_dir)

    if df.empty:
        print("No data found! Exiting.")
        return

    # Display data overview
    print(f"\nDATA OVERVIEW:")
    print(f"Total prediction steps: {len(df)}")
    print(f"Unique models: {df['model_name'].nunique()}")
    print(f"Unique task types: {df['task_type'].nunique()}")
    print(f"Operation type distribution:")
    print(df['operation_type'].value_counts())

    # Calculate accuracy statistics
    stats_df = calculate_accuracy_stats(df)

    if stats_df.empty:
        print("No sufficient data for analysis! Exiting.")
        return

    # Generate visualizations
    create_bar_charts_by_task(stats_df, output_dir)
    create_accuracy_difference_heatmap(stats_df, output_dir)
    create_scatter_plot(stats_df, output_dir)
    create_task_difference_line_plot(stats_df, output_dir)

    # Perform statistical analysis
    analysis_results = perform_statistical_analysis(df, stats_df, output_dir)

    # Create summary files
    create_summary_csv(df, stats_df, analysis_results, output_dir)

    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print("Generated files:")
    print("- accuracy_bars_*.pdf (bar charts by task type)")
    print("- accuracy_difference_heatmap.pdf")
    print("- adding_vs_removing_scatter.pdf")
    print("- task_difference_trends.pdf")
    print("- statistical_analysis_report.txt")
    print("- detailed_accuracy_stats.csv")
    print("- model_summary_stats.csv")

if __name__ == "__main__":
    main()