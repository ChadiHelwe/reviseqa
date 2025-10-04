import json
import os
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for publication-quality figures
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300

# Path to the verified dataset
DATA_DIR = "reviseqa_data/nl/verified"

def load_dataset():
    """Load all non-truncated files"""
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json') and 'truncated' not in f]
    print(f"Found {len(files)} non-truncated files")

    dataset = []
    for filename in files:
        filepath = os.path.join(DATA_DIR, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                dataset.append(data)
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    return dataset

def analyze_modification_patterns(dataset):
    """Analyze what types of edits are made"""
    edit_patterns = defaultdict(lambda: {'count': 0, 'examples': []})

    for ex_idx, example in enumerate(dataset):
        for edit in example.get('edits', []):
            edits_made = edit.get('edits_made', {})

            # Categorize edit pattern
            removed_facts = len(edits_made.get('removed_facts', []))
            removed_rules = len(edits_made.get('removed_rules', []))
            added_facts = len(edits_made.get('added_facts', []))
            added_rules = len(edits_made.get('added_rules', []))

            pattern_key = f"F-{removed_facts}/R-{removed_rules} → F+{added_facts}/R+{added_rules}"
            edit_patterns[pattern_key]['count'] += 1

            if len(edit_patterns[pattern_key]['examples']) < 3:
                edit_patterns[pattern_key]['examples'].append(ex_idx)

    return edit_patterns

def create_visualizations(dataset, output_dir='analysis_output'):
    """Create visualizations for paper"""
    os.makedirs(output_dir, exist_ok=True)

    # Extract data for visualizations
    contexts_length = [len(ex['original_context']) for ex in dataset]
    num_predicates = []
    reasoning_lengths = [len(ex['reasoning_chain']) for ex in dataset]

    for ex in dataset:
        predicates = set()
        for fol in ex['original_context_fol']:
            import re
            preds = re.findall(r'p_\d+', fol)
            predicates.update(preds)
        num_predicates.append(len(predicates))

    # Figure 1: Distribution of key metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(contexts_length, bins=range(min(contexts_length), max(contexts_length)+2),
                 color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Number of Context Statements', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Context Length Distribution', fontsize=13, fontweight='bold')
    axes[0].axvline(np.mean(contexts_length), color='red', linestyle='--',
                    label=f'Mean = {np.mean(contexts_length):.1f}')
    axes[0].legend()

    axes[1].hist(num_predicates, bins=range(min(num_predicates), max(num_predicates)+2),
                 color='forestgreen', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Number of Unique Predicates', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Predicate Count Distribution', fontsize=13, fontweight='bold')
    axes[1].axvline(np.mean(num_predicates), color='red', linestyle='--',
                    label=f'Mean = {np.mean(num_predicates):.1f}')
    axes[1].legend()

    axes[2].hist(reasoning_lengths, bins=range(min(reasoning_lengths), max(reasoning_lengths)+2),
                 color='darkorange', edgecolor='black', alpha=0.7)
    axes[2].set_xlabel('Reasoning Chain Length', fontsize=12)
    axes[2].set_ylabel('Frequency', fontsize=12)
    axes[2].set_title('Reasoning Steps Distribution', fontsize=13, fontweight='bold')
    axes[2].axvline(np.mean(reasoning_lengths), color='red', linestyle='--',
                    label=f'Mean = {np.mean(reasoning_lengths):.1f}')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/dataset_distributions.pdf', bbox_inches='tight')
    plt.savefig(f'{output_dir}/dataset_distributions.png', bbox_inches='tight', dpi=300)
    print(f"✅ Saved: {output_dir}/dataset_distributions.pdf")
    plt.close()

    # Figure 2: Modification type breakdown
    mod_types = Counter()
    for ex in dataset:
        for edit in ex.get('edits', []):
            mod_types[edit.get('modification_type', 'UNKNOWN')] += 1

    fig, ax = plt.subplots(figsize=(8, 6))
    labels = list(mod_types.keys())
    sizes = list(mod_types.values())
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                       colors=colors[:len(labels)], startangle=90,
                                       textprops={'fontsize': 12})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax.set_title('Modification Type Distribution\n(Total: {} edits)'.format(sum(sizes)),
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/modification_types.pdf', bbox_inches='tight')
    plt.savefig(f'{output_dir}/modification_types.png', bbox_inches='tight', dpi=300)
    print(f"✅ Saved: {output_dir}/modification_types.pdf")
    plt.close()

    # Figure 3: Model verification performance
    model_results = defaultdict(lambda: {'verified': 0, 'total': 0})

    for ex in dataset:
        for edit in ex.get('edits', []):
            if 'model_results' in edit:
                for model, results in edit['model_results'].items():
                    for result in results:
                        model_results[model]['total'] += 1
                        if result.get('verified', False):
                            model_results[model]['verified'] += 1

    models = list(model_results.keys())
    # Clean model names for display
    model_display_names = [m.split('/')[-1] if '/' in m else m for m in models]
    verification_rates = [100 * model_results[m]['verified'] / model_results[m]['total']
                         for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(model_display_names, verification_rates, color=['#2ecc71', '#3498db', '#e74c3c'])
    ax.set_ylabel('Verification Rate (%)', fontsize=13)
    ax.set_xlabel('Model', fontsize=13)
    ax.set_title('Model Verification Performance', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)

    # Add value labels on bars
    for i, (bar, rate) in enumerate(zip(bars, verification_rates)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%\n({model_results[models[i]]["verified"]}/{model_results[models[i]]["total"]})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_verification.pdf', bbox_inches='tight')
    plt.savefig(f'{output_dir}/model_verification.png', bbox_inches='tight', dpi=300)
    print(f"✅ Saved: {output_dir}/model_verification.pdf")
    plt.close()

    # Figure 4: Edit operation heatmap
    edit_operations = []
    for ex in dataset:
        for edit in ex.get('edits', []):
            edits_made = edit.get('edits_made', {})
            edit_operations.append({
                'Facts Removed': len(edits_made.get('removed_facts', [])),
                'Rules Removed': len(edits_made.get('removed_rules', [])),
                'Facts Added': len(edits_made.get('added_facts', [])),
                'Rules Added': len(edits_made.get('added_rules', []))
            })

    df_ops = pd.DataFrame(edit_operations)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_ops.corr(), annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, ax=ax, fmt='.2f',
                cbar_kws={'label': 'Correlation'})
    ax.set_title('Edit Operation Correlations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/edit_correlations.pdf', bbox_inches='tight')
    plt.savefig(f'{output_dir}/edit_correlations.png', bbox_inches='tight', dpi=300)
    print(f"✅ Saved: {output_dir}/edit_correlations.pdf")
    plt.close()

def generate_latex_table(dataset, output_dir='analysis_output'):
    """Generate LaTeX table for paper"""

    # Calculate statistics
    contexts_length = [len(ex['original_context']) for ex in dataset]
    num_predicates = []
    reasoning_lengths = [len(ex['reasoning_chain']) for ex in dataset]
    num_edits = [len(ex.get('edits', [])) for ex in dataset]

    for ex in dataset:
        predicates = set()
        for fol in ex['original_context_fol']:
            import re
            preds = re.findall(r'p_\d+', fol)
            predicates.update(preds)
        num_predicates.append(len(predicates))

    # Answer distribution
    answers = Counter([ex['answer'] for ex in dataset])

    # Total edits
    total_edits = sum(num_edits)
    mod_types = Counter()
    for ex in dataset:
        for edit in ex.get('edits', []):
            mod_types[edit.get('modification_type', 'UNKNOWN')] += 1

    latex_table = r"""\begin{table}[t]
\centering
\caption{ReviseQA Dataset Statistics (Non-Truncated Examples)}
\label{tab:dataset_stats}
\begin{tabular}{lr}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
\multicolumn{2}{l}{\textit{Dataset Size}} \\
Total Examples & """ + f"{len(dataset):,}" + r""" \\
Total Edits & """ + f"{total_edits:,}" + r""" \\
Total Verification Tasks & """ + f"{total_edits * 3:,}" + r""" \\
\midrule
\multicolumn{2}{l}{\textit{Original Examples}} \\
True Conclusions & """ + f"{answers['True']} ({100*answers['True']/len(dataset):.1f}\%)" + r""" \\
False Conclusions & """ + f"{answers['False']} ({100*answers['False']/len(dataset):.1f}\%)" + r""" \\
Uncertain & """ + f"{answers.get('Uncertain', 0)}" + r""" \\
\midrule
\multicolumn{2}{l}{\textit{Context Complexity (Mean ± SD)}} \\
Context Statements & """ + f"{np.mean(contexts_length):.2f} ± {np.std(contexts_length):.2f}" + r""" \\
Unique Predicates & """ + f"{np.mean(num_predicates):.2f} ± {np.std(num_predicates):.2f}" + r""" \\
Reasoning Steps & """ + f"{np.mean(reasoning_lengths):.2f} ± {np.std(reasoning_lengths):.2f}" + r""" \\
\midrule
\multicolumn{2}{l}{\textit{Modification Types}} \\
FLIP Edits & """ + f"{mod_types['FLIP']} ({100*mod_types['FLIP']/total_edits:.1f}\%)" + r""" \\
INVARIANT Edits & """ + f"{mod_types['INVARIANT']} ({100*mod_types['INVARIANT']/total_edits:.1f}\%)" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""

    # Save LaTeX table
    with open(f'{output_dir}/dataset_table.tex', 'w') as f:
        f.write(latex_table)

    print(f"✅ Saved: {output_dir}/dataset_table.tex")

    # Also create a model performance table
    model_results = defaultdict(lambda: {'verified': 0, 'total': 0})

    for ex in dataset:
        for edit in ex.get('edits', []):
            if 'model_results' in edit:
                for model, results in edit['model_results'].items():
                    for result in results:
                        model_results[model]['total'] += 1
                        if result.get('verified', False):
                            model_results[model]['verified'] += 1

    model_latex = r"""\begin{table}[t]
\centering
\caption{Model Verification Performance on ReviseQA}
\label{tab:model_performance}
\begin{tabular}{lrr}
\toprule
\textbf{Model} & \textbf{Verified/Total} & \textbf{Accuracy (\%)} \\
\midrule
"""

    for model in sorted(model_results.keys()):
        model_name = model.split('/')[-1] if '/' in model else model
        verified = model_results[model]['verified']
        total = model_results[model]['total']
        rate = 100 * verified / total
        model_latex += f"{model_name} & {verified:,}/{total:,} & {rate:.1f} \\\\\n"

    model_latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(f'{output_dir}/model_performance_table.tex', 'w') as f:
        f.write(model_latex)

    print(f"✅ Saved: {output_dir}/model_performance_table.tex")

def generate_paper_summary(dataset, output_dir='analysis_output'):
    """Generate a summary text file for easy reference"""

    summary = []
    summary.append("="*80)
    summary.append("REVISEQA DATASET ANALYSIS - FOR ACL PAPER")
    summary.append("="*80)
    summary.append("")

    # Key statistics
    summary.append("KEY STATISTICS FOR ABSTRACT/INTRODUCTION:")
    summary.append(f"  - {len(dataset)} verified examples with {sum([len(ex.get('edits', [])) for ex in dataset])} total edits")

    contexts_length = [len(ex['original_context']) for ex in dataset]
    reasoning_lengths = [len(ex['reasoning_chain']) for ex in dataset]

    summary.append(f"  - Average {np.mean(contexts_length):.1f} context statements per example")
    summary.append(f"  - Average {np.mean(reasoning_lengths):.1f} reasoning steps per example")

    mod_types = Counter()
    for ex in dataset:
        for edit in ex.get('edits', []):
            mod_types[edit.get('modification_type', 'UNKNOWN')] += 1

    summary.append(f"  - {mod_types['FLIP']} FLIP edits and {mod_types['INVARIANT']} INVARIANT edits")

    # Model performance
    summary.append("")
    summary.append("MODEL PERFORMANCE HIGHLIGHTS:")

    model_results = defaultdict(lambda: {'verified': 0, 'total': 0})
    for ex in dataset:
        for edit in ex.get('edits', []):
            if 'model_results' in edit:
                for model, results in edit['model_results'].items():
                    for result in results:
                        model_results[model]['total'] += 1
                        if result.get('verified', False):
                            model_results[model]['verified'] += 1

    for model in sorted(model_results.keys(), key=lambda x: model_results[x]['verified']/model_results[x]['total'], reverse=True):
        rate = 100 * model_results[model]['verified'] / model_results[model]['total']
        summary.append(f"  - {model}: {rate:.1f}% verification rate")

    # Dataset balance
    answers = Counter([ex['answer'] for ex in dataset])
    summary.append("")
    summary.append("DATASET BALANCE:")
    summary.append(f"  - True conclusions: {answers['True']} ({100*answers['True']/len(dataset):.1f}%)")
    summary.append(f"  - False conclusions: {answers['False']} ({100*answers['False']/len(dataset):.1f}%)")
    summary.append(f"  - Uncertain: {answers.get('Uncertain', 0)}")

    summary.append("")
    summary.append("="*80)

    summary_text = "\n".join(summary)

    with open(f'{output_dir}/paper_summary.txt', 'w') as f:
        f.write(summary_text)

    print(f"✅ Saved: {output_dir}/paper_summary.txt")
    print("\n" + summary_text)

if __name__ == "__main__":
    output_dir = 'analysis_output'
    os.makedirs(output_dir, exist_ok=True)

    print("Loading dataset...")
    dataset = load_dataset()

    print("\nCreating visualizations...")
    create_visualizations(dataset, output_dir)

    print("\nGenerating LaTeX tables...")
    generate_latex_table(dataset, output_dir)

    print("\nGenerating paper summary...")
    generate_paper_summary(dataset, output_dir)

    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  📊 Visualizations: dataset_distributions.pdf, modification_types.pdf, model_verification.pdf")
    print("  📝 LaTeX Tables: dataset_table.tex, model_performance_table.tex")
    print("  📄 Summary: paper_summary.txt")
    print("  📈 Data: dataset_statistics.csv")
