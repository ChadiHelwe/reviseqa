import json
import os
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

DATA_DIR = "reviseqa_data/nl/verified"

def load_dataset():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json') and 'truncated' not in f]
    dataset = []
    for filename in files:
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, 'r') as f:
            dataset.append(json.load(f))
    return dataset

def analyze_edit_patterns(dataset):
    """Detailed analysis of edit patterns"""
    print("\n" + "="*80)
    print("EDIT PATTERN ANALYSIS")
    print("="*80)

    # Track patterns
    edit_type_patterns = defaultdict(lambda: {'flip': 0, 'invariant': 0})

    for ex in dataset:
        for edit in ex.get('edits', []):
            mod_type = edit.get('modification_type', 'UNKNOWN')
            edits_made = edit.get('edits_made', {})

            # Create pattern signature
            removed_facts = len(edits_made.get('removed_facts', []))
            removed_rules = len(edits_made.get('removed_rules', []))
            added_facts = len(edits_made.get('added_facts', []))
            added_rules = len(edits_made.get('added_rules', []))

            pattern = f"R_F:{removed_facts} R_R:{removed_rules} A_F:{added_facts} A_R:{added_rules}"

            if mod_type == 'FLIP':
                edit_type_patterns[pattern]['flip'] += 1
            elif mod_type == 'INVARIANT':
                edit_type_patterns[pattern]['invariant'] += 1

    # Print most common patterns
    print("\nTop 10 Most Common Edit Patterns:")
    print(f"{'Pattern':<40} {'FLIP':<10} {'INVARIANT':<10} {'Total':<10}")
    print("-" * 80)

    sorted_patterns = sorted(edit_type_patterns.items(),
                           key=lambda x: x[1]['flip'] + x[1]['invariant'],
                           reverse=True)

    for pattern, counts in sorted_patterns[:10]:
        total = counts['flip'] + counts['invariant']
        print(f"{pattern:<40} {counts['flip']:<10} {counts['invariant']:<10} {total:<10}")

def analyze_logical_operators(dataset):
    """Analyze FOL operator usage"""
    print("\n" + "="*80)
    print("LOGICAL OPERATOR USAGE")
    print("="*80)

    import re

    operator_counts = Counter()

    for ex in dataset:
        for fol in ex['original_context_fol']:
            # Count logical operators
            operator_counts['∀ (forall)'] += len(re.findall(r'∀', fol))
            operator_counts['∃ (exists)'] += len(re.findall(r'∃', fol))
            operator_counts['→ (implies)'] += len(re.findall(r'→', fol))
            operator_counts['∧ (and)'] += len(re.findall(r'∧', fol))
            operator_counts['∨ (or)'] += len(re.findall(r'∨', fol))
            operator_counts['¬ (not)'] += len(re.findall(r'¬', fol))
            operator_counts['⊕ (xor)'] += len(re.findall(r'⊕', fol))

    print("\nOperator Frequency:")
    for op, count in operator_counts.most_common():
        print(f"  {op}: {count:,}")

def analyze_reasoning_complexity(dataset):
    """Analyze reasoning chain complexity"""
    print("\n" + "="*80)
    print("REASONING CHAIN COMPLEXITY")
    print("="*80)

    # Types of reasoning steps
    steps_with_conclusion = []
    steps_without_conclusion = []
    avg_facts_per_step = []
    avg_rules_per_step = []

    for ex in dataset:
        for step in ex['reasoning_chain']:
            num_facts = len(step.get('facts', []))
            num_rules = len(step.get('rules', []))

            avg_facts_per_step.append(num_facts)
            avg_rules_per_step.append(num_rules)

            if step.get('conclusion') is not None:
                steps_with_conclusion.append(step)
            else:
                steps_without_conclusion.append(step)

    print(f"\nReasoning Steps Analysis:")
    print(f"  Total reasoning steps: {len(avg_facts_per_step):,}")
    print(f"  Steps with conclusions: {len(steps_with_conclusion):,} ({100*len(steps_with_conclusion)/len(avg_facts_per_step):.1f}%)")
    print(f"  Steps without conclusions: {len(steps_without_conclusion):,} ({100*len(steps_without_conclusion)/len(avg_facts_per_step):.1f}%)")
    print(f"\n  Average facts per step: {np.mean(avg_facts_per_step):.2f} (±{np.std(avg_facts_per_step):.2f})")
    print(f"  Average rules per step: {np.mean(avg_rules_per_step):.2f} (±{np.std(avg_rules_per_step):.2f})")

def analyze_answer_flips(dataset):
    """Analyze how often answers flip during editing"""
    print("\n" + "="*80)
    print("ANSWER FLIP ANALYSIS")
    print("="*80)

    flip_stats = {'total_edits': 0, 'answer_flips': 0, 'answer_preserved': 0}
    flip_types = Counter()

    for ex in dataset:
        original_answer = ex['answer']

        for edit in ex.get('edits', []):
            flip_stats['total_edits'] += 1
            edit_answer = edit.get('answer', 'Unknown')

            if edit_answer != original_answer:
                flip_stats['answer_flips'] += 1
                flip_types[f"{original_answer} → {edit_answer}"] += 1
            else:
                flip_stats['answer_preserved'] += 1

    print(f"\nAnswer Consistency:")
    print(f"  Total edits analyzed: {flip_stats['total_edits']:,}")
    print(f"  Answer flipped: {flip_stats['answer_flips']:,} ({100*flip_stats['answer_flips']/flip_stats['total_edits']:.1f}%)")
    print(f"  Answer preserved: {flip_stats['answer_preserved']:,} ({100*flip_stats['answer_preserved']/flip_stats['total_edits']:.1f}%)")

    print(f"\nMost common answer transitions:")
    for transition, count in flip_types.most_common(5):
        print(f"  {transition}: {count:,}")

def analyze_verification_errors(dataset):
    """Analyze patterns in verification failures"""
    print("\n" + "="*80)
    print("VERIFICATION ERROR ANALYSIS")
    print("="*80)

    model_errors = defaultdict(list)

    for ex_idx, ex in enumerate(dataset):
        for edit_idx, edit in enumerate(ex.get('edits', [])):
            if 'model_results' in edit:
                for model, results in edit['model_results'].items():
                    for result in results:
                        if not result.get('verified', False):
                            mistake = result.get('mistake', 'No mistake info')
                            model_errors[model].append({
                                'example': ex_idx,
                                'edit': edit_idx,
                                'mistake': mistake[:200]  # First 200 chars
                            })

    print("\nVerification Failures by Model:")
    for model, errors in sorted(model_errors.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"\n  {model}: {len(errors)} failures")
        if errors:
            print(f"    Sample error: {errors[0]['mistake'][:150]}...")

def create_detailed_csv_export(dataset, output_dir='analysis_output'):
    """Create detailed CSV for further analysis"""

    rows = []

    for ex_idx, ex in enumerate(dataset):
        # Example-level info
        ex_data = {
            'example_id': ex_idx,
            'original_answer': ex['answer'],
            'num_context_statements': len(ex['original_context']),
            'num_reasoning_steps': len(ex['reasoning_chain']),
            'conclusion': ex['conclusion']
        }

        # Count predicates
        import re
        predicates = set()
        for fol in ex['original_context_fol']:
            preds = re.findall(r'p_\d+', fol)
            predicates.update(preds)
        ex_data['num_unique_predicates'] = len(predicates)

        # Edit-level info
        for edit_idx, edit in enumerate(ex.get('edits', [])):
            edit_data = ex_data.copy()
            edit_data['edit_id'] = edit_idx
            edit_data['edit_number'] = edit.get('edit_number', edit_idx + 1)
            edit_data['modification_type'] = edit.get('modification_type', 'UNKNOWN')
            edit_data['edit_answer'] = edit.get('answer', 'Unknown')

            edits_made = edit.get('edits_made', {})
            edit_data['removed_facts'] = len(edits_made.get('removed_facts', []))
            edit_data['removed_rules'] = len(edits_made.get('removed_rules', []))
            edit_data['added_facts'] = len(edits_made.get('added_facts', []))
            edit_data['added_rules'] = len(edits_made.get('added_rules', []))

            # Model verification
            if 'model_results' in edit:
                for model, results in edit['model_results'].items():
                    clean_model_name = model.split('/')[-1] if '/' in model else model
                    for result in results:
                        edit_data[f'{clean_model_name}_verified'] = result.get('verified', False)

            rows.append(edit_data)

    df = pd.DataFrame(rows)
    df.to_csv(f'{output_dir}/detailed_dataset_export.csv', index=False)
    print(f"\n✅ Detailed export saved to: {output_dir}/detailed_dataset_export.csv")

    return df

if __name__ == "__main__":
    print("Loading dataset...")
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} examples\n")

    analyze_edit_patterns(dataset)
    analyze_logical_operators(dataset)
    analyze_reasoning_complexity(dataset)
    analyze_answer_flips(dataset)
    analyze_verification_errors(dataset)

    print("\n" + "="*80)
    print("Creating detailed CSV export...")
    print("="*80)
    df = create_detailed_csv_export(dataset)

    print("\n" + "="*80)
    print("✅ DETAILED ANALYSIS COMPLETE!")
    print("="*80)
