#!/usr/bin/env python3
"""
LaTeX Table Converter for LCATA Scores
Converts CSV results to a formatted LaTeX table with confidence intervals
"""

import pandas as pd
import re
from collections import defaultdict


def extract_model_family(model_name):
    """Extract model family from model name."""
    # Handle provider/model format
    if '/' in model_name:
        provider, model = model_name.split('/', 1)
        # Extract base model name before version numbers
        base_model = re.split(r'[-_]\d', model)[0]
        return f"{provider}-{base_model}"

    # Handle direct model names
    if 'gemini' in model_name.lower():
        return 'Google-Gemini'
    elif 'gemma' in model_name.lower():
        return 'Google-Gemma'
    elif 'claude' in model_name.lower():
        return 'Anthropic-Claude'
    elif 'gpt' in model_name.lower():
        return 'OpenAI-GPT'
    elif 'qwen' in model_name.lower():
        return 'Alibaba-Qwen'
    elif 'grok' in model_name.lower():
        return 'xAI-Grok'
    elif 'kimi' in model_name.lower():
        return 'Moonshot-Kimi'
    else:
        # Fallback: use first part of model name
        return model_name.split('-')[0].capitalize()


def format_score_with_ci(score, lower_bound, upper_bound):
    """Format score with confidence interval."""
    return f"{score:.3f} ({lower_bound:.3f}, {upper_bound:.3f})"


def clean_model_name(model_name):
    """
    Clean up model names by removing hyphens and trailing numbers/dates.

    Args:
        model_name: Original model name

    Returns:
        Cleaned model name
    """
    # Remove common suffixes and trailing numbers/dates
    cleaned = model_name

    # Remove trailing numbers/dates like -2507, -0905, etc.
    cleaned = re.sub(r'-\d{3,4}$', '', cleaned)

    # Remove specific version patterns
    cleaned = re.sub(r'-a\d+b$', '', cleaned)  # Remove -a22b, -a3b

    # Replace hyphens with spaces for better readability
    cleaned = cleaned.replace('-', ' ')

    # Clean up specific patterns
    cleaned = re.sub(r'\s+', ' ', cleaned)  # Remove multiple spaces
    cleaned = cleaned.strip()

    return cleaned


def get_base_task(task_name):
    """Extract the base task name (explicit, implicit, etc.) from full task name."""
    # Remove _no_reasoning and _no_correction variants to get base task
    base_task = task_name.replace('_no_reasoning', '').replace('_no_correction', '')
    return base_task


def is_cot_task(task_name):
    """Determine if task uses COT (Chain of Thought) or Standard reasoning."""
    # COT: tasks without 'no_reasoning'
    # Standard: tasks with 'no_reasoning'
    return 'no_reasoning' not in task_name


def get_task_variants(base_task):
    """Get all variants of a base task."""
    if base_task.endswith('_no_correction'):
        # For no_correction tasks, insert _no_reasoning before _no_correction
        base_without_correction = base_task.replace('_no_correction', '')
        variants = {
            'cot': base_task,  # e.g., implicit_no_correction
            'standard': f"{base_without_correction}_no_reasoning_no_correction"  # e.g., implicit_no_reasoning_no_correction
        }
    else:
        # For regular tasks
        variants = {
            'cot': base_task,  # e.g., implicit
            'standard': f"{base_task}_no_reasoning"  # e.g., implicit_no_reasoning
        }
    return variants


def determine_task_type(task_name):
    """Determine task type from task name."""
    # Handle no_correction variants
    if 'no_correction' in task_name:
        if task_name.startswith('explicit'):
            return 'explicit_no_correction'
        elif task_name.startswith('implicit'):
            return 'implicit_no_correction'
    else:
        # Handle standard variants
        if task_name.startswith('explicit'):
            return 'explicit'
        elif task_name.startswith('implicit'):
            return 'implicit'

    return 'unknown'


def determine_reasoning_type(task_name):
    """Determine if task uses reasoning or no reasoning."""
    if 'no_reasoning' in task_name:
        return 'no_reasoning'
    else:
        return 'reasoning'


def get_color_for_rank(rank, total_models, max_intensity=0.90):
    """
    Get a color based on rank, with best performing models getting darker colors.

    Args:
        rank: 1-based rank (1 = best)
        total_models: Total number of models
        max_intensity: Maximum color intensity (0-1, lower = more readable)

    Returns:
        LaTeX color command
    """
    if rank == 1:
        # Best model gets the darkest color, but not too dark
        opacity = max_intensity
    else:
        # Linear gradient from max_intensity to 0.05
        opacity = max(0.05, max_intensity * (1 - (rank - 1) / (total_models - 1)))

    # Use a blue gradient for better visibility
    return f"\\cellcolor{{green!{int(opacity * 100)}}}"


def rank_scores_by_column(organized_data):
    """
    Rank scores for each column (K value and reasoning type combination).

    Returns:
        Dictionary with rankings for each column
    """
    rankings = {}

    # For each K value and reasoning type combination
    for k in [2, 4, 7]:
        for reasoning_type in ['COT', 'Standard']:
            column_key = f"k{k}_{reasoning_type}"

            # Collect all scores for this column
            scores_and_models = []
            for model in organized_data:
                if k in organized_data[model] and reasoning_type in organized_data[model][k]:
                    score = organized_data[model][k][reasoning_type]['score']
                    scores_and_models.append((score, model))

            # Sort by score (descending - higher is better)
            scores_and_models.sort(reverse=True)

            # Create ranking dictionary for this column
            rankings[column_key] = {}
            for rank, (score, model) in enumerate(scores_and_models, 1):
                rankings[column_key][model] = rank

    return rankings


def create_latex_table(csv_file='lcata_scores.csv', output_file='lcata_table.tex'):
    """
    Create a LaTeX table from LCATA scores CSV file.

    Args:
        csv_file: Path to the CSV file with LCATA scores
        output_file: Path to output LaTeX file
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Create a nested dictionary to organize data
    # Structure: model -> k_value -> task_type -> reasoning_type -> data
    organized_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

    for _, row in df.iterrows():
        model = row['model']
        task = row['task']
        k = row['k']
        score = row['score']
        lower_bound = row['lower_bound']
        upper_bound = row['upper_bound']

        task_type = determine_task_type(task)
        reasoning_type = determine_reasoning_type(task)

        organized_data[model][k][task_type][reasoning_type] = {
            'score': score,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }

    # Get unique models and sort them
    models = sorted(organized_data.keys())

    # Start building LaTeX table
    latex_content = []
    latex_content.append("\\begin{table}[h!]")
    latex_content.append("\\centering")
    latex_content.append("\\caption{LCATA Scores with 95\\% Confidence Intervals}")
    latex_content.append("\\label{tab:lcata_scores}")
    latex_content.append("\\resizebox{\\textwidth}{!}{%")
    latex_content.append("\\begin{tabular}{lcccccc}")
    latex_content.append("\\toprule")

    # Header
    header = []
    header.append("\\multirow{3}{*}{\\textbf{Model}} & ")
    header.append("\\multicolumn{2}{c}{\\textbf{Easy (K=2)}} & ")
    header.append("\\multicolumn{2}{c}{\\textbf{Medium (K=4)}} & ")
    header.append("\\multicolumn{2}{c}{\\textbf{Hard (K=7)}} \\\\")
    latex_content.append("".join(header))

    latex_content.append("\\cmidrule{2-7}")

    # Sub-header
    subheader = []
    subheader.append(" & ")
    subheader.append("\\textbf{COT} & \\textbf{Standard} & ")
    subheader.append("\\textbf{COT} & \\textbf{Standard} & ")
    subheader.append("\\textbf{COT} & \\textbf{Standard} \\\\")
    latex_content.append("".join(subheader))

    latex_content.append("\\midrule")

    # Data rows
    for model in models:
        # Create row for reasoning tasks
        reasoning_row = []
        reasoning_row.append(f"\\multirow{{2}}{{*}}{{{model}}} & ")

        # For each k value (2, 4, 7)
        for k in [2, 4, 7]:
            # COT (implicit) with reasoning
            cot_data = organized_data[model][k]['COT']['reasoning']
            if cot_data:
                cot_score = format_score_with_ci(
                    cot_data['score'],
                    cot_data['lower_bound'],
                    cot_data['upper_bound']
                )
            else:
                cot_score = "N/A"

            # Standard (explicit) with reasoning
            std_data = organized_data[model][k]['Standard']['reasoning']
            if std_data:
                std_score = format_score_with_ci(
                    std_data['score'],
                    std_data['lower_bound'],
                    std_data['upper_bound']
                )
            else:
                std_score = "N/A"

            reasoning_row.append(f"{cot_score} & {std_score}")
            if k < 7:  # Add separator except for last column
                reasoning_row.append(" & ")

        reasoning_row.append(" \\\\")
        latex_content.append("".join(reasoning_row))

        # Create row for no-reasoning tasks
        no_reasoning_row = []
        no_reasoning_row.append(" & ")  # Empty model cell due to multirow

        # For each k value (2, 4, 7)
        for k in [2, 4, 7]:
            # COT (implicit) without reasoning
            cot_data = organized_data[model][k]['COT']['no_reasoning']
            if cot_data:
                cot_score = format_score_with_ci(
                    cot_data['score'],
                    cot_data['lower_bound'],
                    cot_data['upper_bound']
                )
            else:
                cot_score = "N/A"

            # Standard (explicit) without reasoning
            std_data = organized_data[model][k]['Standard']['no_reasoning']
            if std_data:
                std_score = format_score_with_ci(
                    std_data['score'],
                    std_data['lower_bound'],
                    std_data['upper_bound']
                )
            else:
                std_score = "N/A"

            no_reasoning_row.append(f"{cot_score} & {std_score}")
            if k < 7:  # Add separator except for last column
                no_reasoning_row.append(" & ")

        no_reasoning_row.append(" \\\\")
        latex_content.append("".join(no_reasoning_row))

        if model != models[-1]:  # Add midrule between models except for last one
            latex_content.append("\\midrule")

    # Close table
    latex_content.append("\\bottomrule")
    latex_content.append("\\end{tabular}")
    latex_content.append("}")
    latex_content.append("\\end{table}")

    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex_content))

    print(f"LaTeX table saved to {output_file}")
    return '\n'.join(latex_content)


def create_simplified_latex_table(csv_file='lcata_scores.csv', output_file='lcata_table_simple.tex'):
    """
    Create a simplified LaTeX table with booktabs formatting.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Pivot the data for easier processing
    pivot_data = defaultdict(lambda: defaultdict(dict))

    for _, row in df.iterrows():
        model = row['model']
        task = row['task']
        k = row['k']
        score = row['score']
        lower_bound = row['lower_bound']
        upper_bound = row['upper_bound']

        formatted_score = f"{score:.3f}"
        formatted_ci = f"({lower_bound:.3f}, {upper_bound:.3f})"

        pivot_data[model][f"{task}_k{k}"] = {
            'score': formatted_score,
            'ci': formatted_ci
        }

    # Get sorted models
    models = sorted(pivot_data.keys())

    # Start building LaTeX table
    latex_content = []
    latex_content.append("\\begin{table}[h!]")
    latex_content.append("\\centering")
    latex_content.append("\\caption{LCATA Scores with 95\\% Confidence Intervals}")
    latex_content.append("\\label{tab:lcata_scores}")
    latex_content.append("\\footnotesize")
    latex_content.append("\\begin{tabular}{lcccccc}")
    latex_content.append("\\toprule")

    # Header
    header = []
    header.append("\\textbf{Model} & ")
    header.append("\\textbf{Easy} & \\textbf{Medium} & \\textbf{Hard} & ")
    header.append("\\textbf{Easy} & \\textbf{Medium} & \\textbf{Hard} \\\\")
    latex_content.append("".join(header))

    # Sub-header
    subheader = []
    subheader.append(" & ")
    subheader.append("\\multicolumn{3}{c}{\\textbf{With Reasoning}} & ")
    subheader.append("\\multicolumn{3}{c}{\\textbf{No Reasoning}} \\\\")
    latex_content.append("".join(subheader))

    # Sub-sub-header
    subsubheader = []
    subsubheader.append(" & ")
    subsubheader.append("\\textbf{(K=2)} & \\textbf{(K=4)} & \\textbf{(K=7)} & ")
    subsubheader.append("\\textbf{(K=2)} & \\textbf{(K=4)} & \\textbf{(K=7)} \\\\")
    latex_content.append("".join(subsubheader))

    latex_content.append("\\midrule")

    # Data rows - focusing on explicit tasks for clarity
    for model in models:
        row = []
        row.append(f"{model} & ")

        # With reasoning (explicit tasks)
        tasks_with_reasoning = ['explicit_k2', 'explicit_k4', 'explicit_k7']
        for task in tasks_with_reasoning:
            if task in pivot_data[model]:
                score_ci = f"{pivot_data[model][task]['score']} {pivot_data[model][task]['ci']}"
            else:
                score_ci = "N/A"
            row.append(score_ci)
            if task != 'explicit_k7':
                row.append(" & ")

        row.append(" & ")

        # No reasoning (explicit_no_reasoning tasks)
        tasks_no_reasoning = ['explicit_no_reasoning_k2', 'explicit_no_reasoning_k4', 'explicit_no_reasoning_k7']
        for task in tasks_no_reasoning:
            if task in pivot_data[model]:
                score_ci = f"{pivot_data[model][task]['score']} {pivot_data[model][task]['ci']}"
            else:
                score_ci = "N/A"
            row.append(score_ci)
            if task != 'explicit_no_reasoning_k7':
                row.append(" & ")

        row.append(" \\\\")
        latex_content.append("".join(row))

    # Close table
    latex_content.append("\\bottomrule")
    latex_content.append("\\end{tabular}")
    latex_content.append("\\end{table}")

    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex_content))

    print(f"Simplified LaTeX table saved to {output_file}")
    return '\n'.join(latex_content)


def create_task_specific_latex_table(csv_file='lcata_scores.csv', base_task='explicit', output_file=None, gradient_colors=True):
    """
    Create a LaTeX table for a specific task (e.g., 'explicit') comparing COT vs Standard.

    Args:
        csv_file: Path to the CSV file with LCATA scores
        base_task: Base task name (e.g., 'explicit', 'implicit', 'explicit_no_correction', 'implicit_no_correction')
        output_file: Output LaTeX file path
        gradient_colors: Whether to apply gradient coloring based on performance
    """
    if output_file is None or "None" in output_file:
        output_file = f'lcata_table_{base_task}.tex'

    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Get task variants
    variants = get_task_variants(base_task)
    cot_task = variants['cot']
    standard_task = variants['standard']

    # Filter data for the specific tasks
    task_data = df[df['task'].isin([cot_task, standard_task])]

    if task_data.empty:
        print(f"No data found for tasks: {cot_task}, {standard_task}")
        return None

    # Organize data by model and k value
    organized_data = defaultdict(lambda: defaultdict(dict))

    for _, row in task_data.iterrows():
        model = row['model']
        task = row['task']
        k = row['k']
        score = row['score']
        lower_bound = row['lower_bound']
        upper_bound = row['upper_bound']

        reasoning_type = 'COT' if is_cot_task(task) else 'Standard'
        organized_data[model][k][reasoning_type] = {
            'score': score,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }

    # Get sorted models
    models = sorted(organized_data.keys())

    # Calculate rankings for gradient coloring
    rankings = rank_scores_by_column(organized_data) if gradient_colors else None
    total_models = len(models)

    # Start building LaTeX table
    latex_content = []
    latex_content.append("\\begin{table}[h!]")
    latex_content.append("\\centering")
    latex_content.append(f"\\caption{{LCATA Scores for {base_task.replace('_', ' ').title()} Task with 95\\% Confidence Intervals}}")
    latex_content.append(f"\\label{{tab:lcata_scores_{base_task}}}")

    # Add required packages for coloring
    if gradient_colors:
        latex_content.append("% Requires: \\usepackage{xcolor} and \\usepackage{colortbl} in preamble")

    latex_content.append("\\resizebox{\\textwidth}{!}{%")
    latex_content.append("\\begin{tabular}{lcccccc}")
    latex_content.append("\\toprule")

    # Header with main columns and subcolumns
    header = []
    header.append("\\multirow{2}{*}{\\textbf{Model}} & ")
    header.append("\\multicolumn{2}{c}{\\textbf{Easy (K=2)}} & ")
    header.append("\\multicolumn{2}{c}{\\textbf{Medium (K=4)}} & ")
    header.append("\\multicolumn{2}{c}{\\textbf{Hard (K=7)}} \\\\")
    latex_content.append("".join(header))

    latex_content.append("\\cmidrule(l){2-3} \\cmidrule(l){4-5} \\cmidrule(l){6-7}")

    # Sub-header with COT and Standard
    subheader = []
    subheader.append(" & ")
    subheader.append("\\textbf{COT} & \\textbf{Standard} & ")
    subheader.append("\\textbf{COT} & \\textbf{Standard} & ")
    subheader.append("\\textbf{COT} & \\textbf{Standard} \\\\")
    latex_content.append("".join(subheader))

    latex_content.append("\\midrule")

    # Data rows
    for model in models:
        row = []
        cleaned_model_name = clean_model_name(model)
        row.append(f"{cleaned_model_name} & ")

        # For each difficulty level (K=2, 4, 7), show COT and Standard side by side
        for k in [2, 4, 7]:
            # COT score
            if 'COT' in organized_data[model][k]:
                data = organized_data[model][k]['COT']
                cot_score = format_score_with_ci(data['score'], data['lower_bound'], data['upper_bound'])

                # Add color if gradient_colors is enabled
                if gradient_colors and rankings:
                    column_key = f"k{k}_COT"
                    if column_key in rankings and model in rankings[column_key]:
                        rank = rankings[column_key][model]
                        color = get_color_for_rank(rank, total_models)
                        cot_score = f"{color}{cot_score}"
            else:
                cot_score = "N/A"

            # Standard score
            if 'Standard' in organized_data[model][k]:
                data = organized_data[model][k]['Standard']
                standard_score = format_score_with_ci(data['score'], data['lower_bound'], data['upper_bound'])

                # Add color if gradient_colors is enabled
                if gradient_colors and rankings:
                    column_key = f"k{k}_Standard"
                    if column_key in rankings and model in rankings[column_key]:
                        rank = rankings[column_key][model]
                        color = get_color_for_rank(rank, total_models)
                        standard_score = f"{color}{standard_score}"
            else:
                standard_score = "N/A"

            row.append(f"{cot_score} & {standard_score}")
            if k != 7:  # Add separator except for last column group
                row.append(" & ")

        row.append(" \\\\")
        latex_content.append("".join(row))

    # Close table
    latex_content.append("\\bottomrule")
    latex_content.append("\\end{tabular}")
    latex_content.append("}")  # Close resizebox
    latex_content.append("\\end{table}")

    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex_content))

    print(f"Task-specific LaTeX table saved to {output_file}")
    return '\n'.join(latex_content)


def create_comprehensive_latex_table(csv_file='lcata_scores.csv', output_file='lcata_comprehensive.tex'):
    """
    Create a comprehensive LaTeX table showing all task types with reasoning and no_reasoning variants.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Task types to include
    task_types = ['explicit', 'implicit', 'explicit_no_correction', 'implicit_no_correction']

    # Organize data by model, task type, k value, and reasoning type
    organized_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

    for _, row in df.iterrows():
        model = row['model']
        task = row['task']
        k = row['k']
        score = row['score']
        lower_bound = row['lower_bound']
        upper_bound = row['upper_bound']

        task_type = determine_task_type(task)
        reasoning_type = determine_reasoning_type(task)

        if task_type in task_types:
            organized_data[model][task_type][k][reasoning_type] = {
                'score': score,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }

    # Get sorted models
    models = sorted(organized_data.keys())

    # Start building LaTeX table
    latex_content = []
    latex_content.append("\\begin{table}[h!]")
    latex_content.append("\\centering")
    latex_content.append("\\caption{Comprehensive LCATA Scores with 95\\% Confidence Intervals}")
    latex_content.append("\\label{tab:lcata_comprehensive}")
    latex_content.append("\\footnotesize")
    latex_content.append("\\begin{tabular}{lcccccccccccc}")
    latex_content.append("\\toprule")

    # Multi-level header
    header = []
    header.append("\\multirow{3}{*}{\\textbf{Model}} & ")
    header.append("\\multicolumn{3}{c}{\\textbf{Explicit}} & ")
    header.append("\\multicolumn{3}{c}{\\textbf{Implicit}} & ")
    header.append("\\multicolumn{3}{c}{\\textbf{Explicit No Correction}} & ")
    header.append("\\multicolumn{3}{c}{\\textbf{Implicit No Correction}} \\\\")
    latex_content.append("".join(header))

    latex_content.append("\\cmidrule(l){2-4} \\cmidrule(l){5-7} \\cmidrule(l){8-10} \\cmidrule(l){11-13}")

    # Sub-header for difficulty levels
    subheader = []
    subheader.append(" & ")
    for _ in range(4):  # For each task type
        subheader.append("\\textbf{K=2} & \\textbf{K=4} & \\textbf{K=7} & ")
    subheader[-1] = subheader[-1].rstrip(" & ") + " \\\\"  # Remove last &
    latex_content.append("".join(subheader))

    latex_content.append("\\midrule")

    # Data rows - separate rows for reasoning and no_reasoning
    for model in models:
        cleaned_model_name = clean_model_name(model)

        # Reasoning row
        reasoning_row = []
        reasoning_row.append(f"\\multirow{{2}}{{*}}{{{cleaned_model_name}}} & ")

        for task_type in task_types:
            for k in [2, 4, 7]:
                if 'reasoning' in organized_data[model][task_type][k]:
                    data = organized_data[model][task_type][k]['reasoning']
                    score = format_score_with_ci(data['score'], data['lower_bound'], data['upper_bound'])
                else:
                    score = "N/A"
                reasoning_row.append(score)
                if not (task_type == task_types[-1] and k == 7):  # Not the last cell
                    reasoning_row.append(" & ")

        reasoning_row.append(" \\\\")
        latex_content.append("".join(reasoning_row))

        # No reasoning row
        no_reasoning_row = []
        no_reasoning_row.append(" & ")  # Empty due to multirow

        for task_type in task_types:
            for k in [2, 4, 7]:
                if 'no_reasoning' in organized_data[model][task_type][k]:
                    data = organized_data[model][task_type][k]['no_reasoning']
                    score = format_score_with_ci(data['score'], data['lower_bound'], data['upper_bound'])
                else:
                    score = "N/A"
                no_reasoning_row.append(score)
                if not (task_type == task_types[-1] and k == 7):  # Not the last cell
                    no_reasoning_row.append(" & ")

        no_reasoning_row.append(" \\\\")
        latex_content.append("".join(no_reasoning_row))

        if model != models[-1]:  # Add separator between models except for last
            latex_content.append("\\midrule")

    # Close table
    latex_content.append("\\bottomrule")
    latex_content.append("\\end{tabular}")
    latex_content.append("\\end{table}")

    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex_content))

    print(f"Comprehensive LaTeX table saved to {output_file}")
    return '\n'.join(latex_content)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert LCATA CSV to LaTeX table")
    parser.add_argument("--input", default="lcata_scores.csv", help="Input CSV file")
    parser.add_argument("--output", default="lcata_table.tex", help="Output LaTeX file")
    parser.add_argument("--simple", action="store_true", help="Create simplified table")
    parser.add_argument("--comprehensive", action="store_true", help="Create comprehensive table with all task types")
    parser.add_argument("--task", default="explicit", help="Specific task for task-specific table")

    args = parser.parse_args()

    if args.comprehensive:
        create_comprehensive_latex_table(args.input, args.output)
    elif args.simple:
        create_simplified_latex_table(args.input, args.output)
    else:
        create_task_specific_latex_table(args.input, args.task, args.output)