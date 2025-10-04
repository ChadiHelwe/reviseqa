import json
import os
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

# Path to the verified dataset
DATA_DIR = "reviseqa_data/nl/verified"

# Load all non-truncated files
def load_dataset():
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

def analyze_dataset(dataset):
    stats = {
        'total_examples': len(dataset),
        'reasoning_chain_lengths': [],
        'num_edits': [],
        'modification_types': Counter(),
        'original_answers': Counter(),
        'contexts_length': [],
        'num_predicates': [],
        'edit_details': defaultdict(list)
    }

    for example in dataset:
        # Original answer distribution
        stats['original_answers'][example['answer']] += 1

        # Context length
        stats['contexts_length'].append(len(example['original_context']))

        # Reasoning chain length
        stats['reasoning_chain_lengths'].append(len(example['reasoning_chain']))

        # Count unique predicates in FOL
        predicates = set()
        for fol in example['original_context_fol']:
            # Extract predicates (p_0, p_1, etc.)
            import re
            preds = re.findall(r'p_\d+', fol)
            predicates.update(preds)
        stats['num_predicates'].append(len(predicates))

        # Edit information
        if 'edits' in example:
            stats['num_edits'].append(len(example['edits']))

            for edit in example['edits']:
                mod_type = edit.get('modification_type', 'UNKNOWN')
                stats['modification_types'][mod_type] += 1

                # Track what was added/removed
                edits_made = edit.get('edits_made', {})
                stats['edit_details']['removed_facts_count'].append(len(edits_made.get('removed_facts', [])))
                stats['edit_details']['removed_rules_count'].append(len(edits_made.get('removed_rules', [])))
                stats['edit_details']['added_facts_count'].append(len(edits_made.get('added_facts', [])))
                stats['edit_details']['added_rules_count'].append(len(edits_made.get('added_rules', [])))

                # Answer after edit
                edit_answer = edit.get('answer', 'Unknown')

                # Model verification stats
                if 'model_results' in edit:
                    for model, results in edit['model_results'].items():
                        for result in results:
                            verified = result.get('verified', False)
                            stats['edit_details'][f'{model}_verification'].append(verified)

    return stats

def print_analysis(stats):
    print("\n" + "="*80)
    print("REVISEQA DATASET ANALYSIS - Non-Truncated Examples")
    print("="*80)

    print(f"\n📊 OVERALL STATISTICS")
    print(f"   Total Examples: {stats['total_examples']}")
    print(f"   Total Edits: {sum(stats['num_edits'])}")
    print(f"   Avg Edits per Example: {np.mean(stats['num_edits']):.2f} (±{np.std(stats['num_edits']):.2f})")

    print(f"\n📝 ORIGINAL EXAMPLES")
    print(f"   Original Answer Distribution:")
    for answer, count in stats['original_answers'].items():
        pct = 100 * count / stats['total_examples']
        print(f"      {answer}: {count} ({pct:.1f}%)")

    print(f"\n🔢 CONTEXT STATISTICS")
    print(f"   Avg Context Statements: {np.mean(stats['contexts_length']):.2f} (±{np.std(stats['contexts_length']):.2f})")
    print(f"   Min/Max: {min(stats['contexts_length'])} / {max(stats['contexts_length'])}")
    print(f"   Avg Unique Predicates: {np.mean(stats['num_predicates']):.2f} (±{np.std(stats['num_predicates']):.2f})")

    print(f"\n🧠 REASONING CHAIN")
    print(f"   Avg Chain Length: {np.mean(stats['reasoning_chain_lengths']):.2f} (±{np.std(stats['reasoning_chain_lengths']):.2f})")
    print(f"   Min/Max: {min(stats['reasoning_chain_lengths'])} / {max(stats['reasoning_chain_lengths'])}")

    print(f"\n✏️  MODIFICATION TYPES")
    total_edits = sum(stats['modification_types'].values())
    for mod_type, count in stats['modification_types'].most_common():
        pct = 100 * count / total_edits
        print(f"   {mod_type}: {count} ({pct:.1f}%)")

    print(f"\n🔄 EDIT OPERATIONS (Average per Edit)")
    if stats['edit_details']['removed_facts_count']:
        print(f"   Facts Removed: {np.mean(stats['edit_details']['removed_facts_count']):.2f}")
        print(f"   Rules Removed: {np.mean(stats['edit_details']['removed_rules_count']):.2f}")
        print(f"   Facts Added: {np.mean(stats['edit_details']['added_facts_count']):.2f}")
        print(f"   Rules Added: {np.mean(stats['edit_details']['added_rules_count']):.2f}")

    # Model verification rates
    print(f"\n🤖 MODEL VERIFICATION RATES")
    model_names = [k for k in stats['edit_details'].keys() if 'verification' in k]
    for model_key in sorted(model_names):
        verifications = stats['edit_details'][model_key]
        if verifications:
            model_name = model_key.replace('_verification', '')
            true_count = sum(verifications)
            total = len(verifications)
            rate = 100 * true_count / total
            print(f"   {model_name}: {true_count}/{total} ({rate:.1f}%)")

    print("\n" + "="*80)

def create_detailed_table(dataset):
    """Create a detailed breakdown table for paper"""
    rows = []

    for example in dataset:
        ex_id = example.get('original_context_fol', [''])[0]  # Use as identifier

        base_row = {
            'num_context_statements': len(example['original_context']),
            'num_predicates': len(set([p for fol in example['original_context_fol']
                                      for p in __import__('re').findall(r'p_\d+', fol)])),
            'reasoning_steps': len(example['reasoning_chain']),
            'original_answer': example['answer'],
            'num_edits': len(example.get('edits', []))
        }

        rows.append(base_row)

    df = pd.DataFrame(rows)

    print("\n📋 SUMMARY TABLE (for paper)")
    print("\nDescriptive Statistics:")
    print(df.describe().round(2))

    print("\n\nAnswer Distribution:")
    print(df['original_answer'].value_counts())

    # Save to CSV
    df.to_csv('analysis_output/dataset_statistics.csv', index=False)
    print("\n✅ Detailed statistics saved to: analysis_output/dataset_statistics.csv")

    return df

# Main execution
if __name__ == "__main__":
    # Create output directory
    os.makedirs('analysis_output', exist_ok=True)

    print("Loading dataset...")
    dataset = load_dataset()

    print("Analyzing dataset...")
    stats = analyze_dataset(dataset)

    print_analysis(stats)

    print("\nCreating detailed tables...")
    df = create_detailed_table(dataset)

    # Additional analysis for paper
    print("\n📊 ADDITIONAL METRICS FOR PAPER")
    print(f"   Total reasoning steps across all examples: {sum(stats['reasoning_chain_lengths'])}")
    print(f"   Total predicate-to-NL mappings to verify: {sum(stats['num_edits']) * 3}")  # Assuming 3 models

    print("\n✅ Analysis complete!")
