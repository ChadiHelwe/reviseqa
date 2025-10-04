import glob
import json
import os


def analyze_feedback_impact(folder_path):
    """
    Analyzes recovery after errors: When a model makes a mistake, does it recover
    better on the next step with feedback/correction vs without?
    """

    json_files = glob.glob("detailed_models_results/**/*.json", recursive=True)

    # Track recovery after errors
    models_with_correction = {}
    models_without_correction = {}

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        model_name = data.get('metadata', {}).get('model_name', 'unknown')

        # Determine if this file has correction/feedback
        has_correction = "no_correction" not in json_file

        if model_name not in models_with_correction:
            models_with_correction[model_name] = {'correct': 0, 'total': 0}
        if model_name not in models_without_correction:
            models_without_correction[model_name] = {'correct': 0, 'total': 0}

        found_error = False
        for pred in data.get('predictions', []):
            if pred.get('is_demonstration', False) or pred.get('step', 0) == 0:
                continue

            # If previous step was wrong, check recovery on this step
            if found_error:
                found_error = False

                if has_correction:
                    models_with_correction[model_name]['total'] += 1
                    if pred.get("correct"):
                        models_with_correction[model_name]['correct'] += 1
                else:
                    models_without_correction[model_name]['total'] += 1
                    if pred.get("correct"):
                        models_without_correction[model_name]['correct'] += 1

            # Track if this step is wrong
            if not pred.get("correct"):
                found_error = True

    # Print results
    print("=" * 80)
    print("FEEDBACK IMPACT ANALYSIS: Recovery After Errors")
    print("=" * 80)
    print("Measures: After an error, what % of time does the model recover on next step?")
    print("=" * 80)

    for model_name in sorted(models_with_correction.keys()):
        if model_name == 'unknown':
            continue

        with_corr = models_with_correction.get(model_name, {'correct': 0, 'total': 0})
        without_corr = models_without_correction.get(model_name, {'correct': 0, 'total': 0})

        with_acc = (with_corr['correct'] / with_corr['total']) if with_corr['total'] > 0 else 0
        without_acc = (without_corr['correct'] / without_corr['total']) if without_corr['total'] > 0 else 0

        print(f"\nModel: {model_name}")
        print(f"  With Correction:    {with_acc:.3f} ({with_acc*100:.1f}%) over {with_corr['total']} recoveries")
        print(f"  Without Correction: {without_acc:.3f} ({without_acc*100:.1f}%) over {without_corr['total']} recoveries")
        print(f"  Difference:         {with_acc - without_acc:+.3f} ({'Better' if with_acc > without_acc else 'Worse'} with correction)")
        print("-" * 80)




if __name__ == "__main__":
    analyze_feedback_impact("detailed_models_results")