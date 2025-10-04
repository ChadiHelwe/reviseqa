#!/usr/bin/env python3
"""
Simple analysis: Do models perform better when adding or removing facts/rules?
"""

import json
import glob
import pandas as pd

def main():
    # Load all JSON result files
    print("Loading data...")
    json_files = glob.glob("detailed_models_results/**/*.json", recursive=True)
    print(f"Found {len(json_files)} result files\n")

    # Collect all predictions
    adding_results = []
    removing_results = []

    for file_path in json_files:
        with open(file_path, 'r') as f:
            data = json.load(f)

        model_name = data.get('metadata', {}).get('model_name', 'unknown')

        for pred in data.get('predictions', []):
            # Skip demonstration steps
            if pred.get('is_demonstration', False) or pred.get('step', 0) == 0:
                continue

            tags = set(pred.get('tags', []))
            correct = pred.get('correct')

            if correct is None:
                continue

            # Check if this is adding or removing
            is_adding = bool(tags & {'added_rules', 'added_facts'})
            is_removing = bool(tags & {'removed_rules', 'removed_facts'})

            if is_adding and not is_removing:
                adding_results.append({'model': model_name, 'correct': correct})
            elif is_removing and not is_adding:
                removing_results.append({'model': model_name, 'correct': correct})

    # Convert to dataframes
    adding_df = pd.DataFrame(adding_results)
    removing_df = pd.DataFrame(removing_results)

    print(f"Adding operations: {len(adding_df)} predictions")
    print(f"Removing operations: {len(removing_df)} predictions\n")

    # Calculate overall accuracy
    adding_acc = adding_df['correct'].mean()
    removing_acc = removing_df['correct'].mean()

    print("=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    print(f"Adding accuracy:   {adding_acc:.3f} ({adding_acc*100:.1f}%)")
    print(f"Removing accuracy: {removing_acc:.3f} ({removing_acc*100:.1f}%)")
    print(f"Difference:        {adding_acc - removing_acc:+.3f}")
    print()

    if adding_acc > removing_acc:
        print(f"✓ Models perform BETTER when ADDING ({(adding_acc - removing_acc)*100:.1f}% higher)")
    else:
        print(f"✓ Models perform BETTER when REMOVING ({(removing_acc - adding_acc)*100:.1f}% higher)")
    print()

    # Per-model breakdown
    print("=" * 60)
    print("PER-MODEL BREAKDOWN")
    print("=" * 60)

    all_models = set(adding_df['model'].unique()) & set(removing_df['model'].unique())

    model_stats = []
    for model in sorted(all_models):
        add_acc = adding_df[adding_df['model'] == model]['correct'].mean()
        rem_acc = removing_df[removing_df['model'] == model]['correct'].mean()
        diff = add_acc - rem_acc

        add_count = len(adding_df[adding_df['model'] == model])
        rem_count = len(removing_df[removing_df['model'] == model])

        model_stats.append({
            'model': model.split('/')[-1],
            'adding_acc': add_acc,
            'removing_acc': rem_acc,
            'difference': diff,
            'adding_count': add_count,
            'removing_count': rem_count
        })

    # Sort by absolute difference
    model_stats.sort(key=lambda x: abs(x['difference']), reverse=True)

    print(f"{'Model':<40} {'Adding':>8} {'Removing':>8} {'Diff':>8} {'Better at'}")
    print("-" * 80)

    for stat in model_stats[:20]:  # Top 20 models
        better = "Adding" if stat['difference'] > 0 else "Removing"
        print(f"{stat['model']:<40} {stat['adding_acc']:>8.3f} {stat['removing_acc']:>8.3f} "
              f"{stat['difference']:>+8.3f} {better}")

    print()

    # Summary statistics
    better_at_adding = sum(1 for s in model_stats if s['difference'] > 0)
    better_at_removing = sum(1 for s in model_stats if s['difference'] < 0)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Models better at adding:   {better_at_adding}/{len(model_stats)}")
    print(f"Models better at removing: {better_at_removing}/{len(model_stats)}")
    print()

    # Save results
    df = pd.DataFrame(model_stats)
    df.to_csv('adding_vs_removing_results.csv', index=False)
    print("Results saved to: adding_vs_removing_results.csv")

if __name__ == "__main__":
    main()
