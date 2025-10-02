#!/usr/bin/env python3
"""
Compute error rates for adding vs removing operations across all model results.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

def analyze_results():
    results_dir = Path("/Users/helwec/Desktop/reviseqa/detailed_models_results")

    # Statistics for each operation type
    stats = {
        'added_facts': {'correct': 0, 'incorrect': 0},
        'added_rules': {'correct': 0, 'incorrect': 0},
        'removed_facts': {'correct': 0, 'incorrect': 0},
        'removed_rules': {'correct': 0, 'incorrect': 0},
        'no_change': {'correct': 0, 'incorrect': 0}
    }

    total_files = 0
    total_steps = 0

    # Walk through all JSON files
    for json_file in results_dir.rglob("*.json"):
        total_files += 1

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Process each step in predictions array
            predictions = data.get('predictions', [])
            for step_data in predictions:
                total_steps += 1

                # Get prediction and correct answer
                prediction = step_data.get('prediction', '').strip().lower()
                correct_answer = step_data.get('correct_answer', '').strip().lower()
                tags = step_data.get('tags', [])

                # Determine if correct
                is_correct = (prediction == correct_answer)

                # Categorize by operation type
                operation_found = False

                if 'added_facts' in tags:
                    if is_correct:
                        stats['added_facts']['correct'] += 1
                    else:
                        stats['added_facts']['incorrect'] += 1
                    operation_found = True

                if 'added_rules' in tags:
                    if is_correct:
                        stats['added_rules']['correct'] += 1
                    else:
                        stats['added_rules']['incorrect'] += 1
                    operation_found = True

                if 'removed_facts' in tags:
                    if is_correct:
                        stats['removed_facts']['correct'] += 1
                    else:
                        stats['removed_facts']['incorrect'] += 1
                    operation_found = True

                if 'removed_rules' in tags:
                    if is_correct:
                        stats['removed_rules']['correct'] += 1
                    else:
                        stats['removed_rules']['incorrect'] += 1
                    operation_found = True

                if not operation_found:
                    if is_correct:
                        stats['no_change']['correct'] += 1
                    else:
                        stats['no_change']['incorrect'] += 1

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    # Print results
    print(f"\n{'='*70}")
    print(f"ANALYSIS OF ADDING vs REMOVING OPERATIONS")
    print(f"{'='*70}\n")
    print(f"Total files analyzed: {total_files}")
    print(f"Total steps analyzed: {total_steps}\n")
    print(f"{'='*70}")
    print(f"{'Operation Type':<20} {'Correct':<12} {'Incorrect':<12} {'Total':<12} {'Error Rate':<12}")
    print(f"{'='*70}")

    for op_type, counts in stats.items():
        correct = counts['correct']
        incorrect = counts['incorrect']
        total = correct + incorrect
        error_rate = (incorrect / total * 100) if total > 0 else 0

        print(f"{op_type:<20} {correct:<12} {incorrect:<12} {total:<12} {error_rate:>10.2f}%")

    print(f"{'='*70}\n")

    # Calculate aggregated statistics
    adding_correct = stats['added_facts']['correct'] + stats['added_rules']['correct']
    adding_incorrect = stats['added_facts']['incorrect'] + stats['added_rules']['incorrect']
    adding_total = adding_correct + adding_incorrect
    adding_error_rate = (adding_incorrect / adding_total * 100) if adding_total > 0 else 0

    removing_correct = stats['removed_facts']['correct'] + stats['removed_rules']['correct']
    removing_incorrect = stats['removed_facts']['incorrect'] + stats['removed_rules']['incorrect']
    removing_total = removing_correct + removing_incorrect
    removing_error_rate = (removing_incorrect / removing_total * 100) if removing_total > 0 else 0

    print(f"{'='*70}")
    print(f"AGGREGATED RESULTS")
    print(f"{'='*70}")
    print(f"{'Category':<20} {'Correct':<12} {'Incorrect':<12} {'Total':<12} {'Error Rate':<12}")
    print(f"{'='*70}")
    print(f"{'ADDING (any)':<20} {adding_correct:<12} {adding_incorrect:<12} {adding_total:<12} {adding_error_rate:>10.2f}%")
    print(f"{'REMOVING (any)':<20} {removing_correct:<12} {removing_incorrect:<12} {removing_total:<12} {removing_error_rate:>10.2f}%")
    print(f"{'='*70}\n")

    # Calculate difference
    diff = adding_error_rate - removing_error_rate
    if diff > 0:
        print(f"✗ Models make {diff:.2f}% MORE errors when ADDING compared to REMOVING")
    else:
        print(f"✓ Models make {abs(diff):.2f}% FEWER errors when ADDING compared to REMOVING")

    print()

    # Additional analysis: check for mixed operations
    print(f"{'='*70}")
    print(f"DETAILED BREAKDOWN")
    print(f"{'='*70}\n")

    # Count steps with multiple operation tags
    print("Note: Steps can have multiple tags (e.g., both 'added_facts' and 'removed_rules').")
    print("The totals above count each tag separately, so they may sum to more than total steps.\n")

if __name__ == "__main__":
    analyze_results()
