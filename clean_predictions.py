#!/usr/bin/env python3
"""
Clean qwen3-coder-30b-a3b-instruct predictions by extracting answers properly
and updating only prediction, reasoning, and is_correct fields while preserving
the original JSON structure.
"""

import json
import os
import sys
from pathlib import Path

# Add src to path to import the extraction function
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from src.evaluation import extract_answer_from_prediction


def process_prediction_file(file_path, output_path):
    """
    Process a single prediction JSON file and update only the necessary fields.
    This handles files with multiple prediction steps.

    Args:
        file_path: Path to the original JSON file
        output_path: Path where the cleaned file should be saved
    """
    try:
        # Load the original JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)

        processed_steps = 0
        corrected_predictions = 0
        examples = []

        # Process each prediction step in the file
        if 'predictions' in data:
            for step_data in data['predictions']:
                if 'prediction' in step_data:
                    original_prediction = step_data.get('prediction', '')
                    correct_answer = step_data.get('correct_answer', '')

                    # Extract answer and reasoning using our improved function
                    extracted_answer, extracted_reasoning = extract_answer_from_prediction(original_prediction)

                    # Check if the prediction is correct
                    is_correct = (extracted_answer == correct_answer)

                    # Update ONLY the specific fields, keeping everything else unchanged
                    step_data['prediction'] = extracted_answer
                    step_data['reasoning'] = extracted_reasoning
                    step_data['correct'] = is_correct

                    processed_steps += 1
                    if is_correct:
                        corrected_predictions += 1

                    # Store examples of complex predictions that were processed
                    if len(original_prediction) > 50 and '{' in original_prediction and len(examples) < 2:
                        examples.append({
                            'step': step_data.get('step', 'unknown'),
                            'original_prediction': original_prediction[:150] + "..." if len(original_prediction) > 150 else original_prediction,
                            'extracted_answer': extracted_answer,
                            'correct_answer': correct_answer,
                            'is_correct': is_correct
                        })

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the file with the same structure but updated fields
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        return {
            'file_name': file_path.name,
            'processed_steps': processed_steps,
            'correct_predictions': corrected_predictions,
            'examples': examples
        }

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def process_model_results(model_path=None, output_path=None):
    """
    Process all results for a model and create cleaned versions
    with the same folder structure.

    Args:
        model_path: Path to the model results folder
        output_path: Path to save cleaned results (defaults to clean_{model_folder_name})
    """
    if model_path is None:
        model_path = "detailed_models_results/qwen/qwen3-coder-30b-a3b-instruct"

    input_base = Path(model_path)

    if output_path is None:
        # Auto-generate output path as clean_{folder_name}
        model_folder_name = input_base.name
        output_path = f"clean_{model_folder_name}"

    output_base = Path(output_path)

    if not input_base.exists():
        print(f"Input directory {input_base} does not exist!")
        return

    # Create output base directory
    output_base.mkdir(exist_ok=True)

    stats = {
        'total_files': 0,
        'processed_files': 0,
        'correct_predictions': 0,
        'incorrect_predictions': 0,
        'tasks': {}
    }

    # Process each task directory (explicit, implicit, etc.)
    for task_dir in input_base.iterdir():
        if task_dir.is_dir():
            task_name = task_dir.name
            print(f"Processing task: {task_name}")

            # Create corresponding output directory with same structure
            output_task_dir = output_base / task_name
            output_task_dir.mkdir(exist_ok=True)

            task_stats = {
                'total_files': 0,
                'processed_files': 0,
                'total_steps': 0,
                'correct_steps': 0,
                'examples': []
            }

            # Process all JSON files in the task directory
            for json_file in task_dir.glob("*.json"):
                stats['total_files'] += 1
                task_stats['total_files'] += 1

                # Keep the same filename in the output directory
                output_file = output_task_dir / json_file.name

                result = process_prediction_file(json_file, output_file)

                if result:
                    stats['processed_files'] += 1
                    task_stats['processed_files'] += 1

                    task_stats['total_steps'] += result['processed_steps']
                    task_stats['correct_steps'] += result['correct_predictions']

                    stats['correct_predictions'] += result['correct_predictions']
                    stats['incorrect_predictions'] += (result['processed_steps'] - result['correct_predictions'])

                    # Store examples from complex predictions
                    task_stats['examples'].extend(result['examples'])

            stats['tasks'][task_name] = task_stats
            accuracy = task_stats['correct_steps'] / max(1, task_stats['total_steps']) * 100
            print(f"  {task_name}: {task_stats['correct_steps']}/{task_stats['total_steps']} steps correct ({accuracy:.1f}%)")

            # Show examples for this task
            for i, example in enumerate(task_stats['examples'][:3], 1):
                print(f"    Example {i} (step {example['step']}):")
                print(f"      Original: {example['original_prediction']}")
                print(f"      Extracted: {example['extracted_answer']}")
                print(f"      Expected: {example['correct_answer']}")
                print(f"      Correct: {example['is_correct']}")

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total files: {stats['total_files']}")
    print(f"Processed files: {stats['processed_files']}")
    print(f"Correct predictions: {stats['correct_predictions']}")
    print(f"Incorrect predictions: {stats['incorrect_predictions']}")

    total_steps = stats['correct_predictions'] + stats['incorrect_predictions']
    if total_steps > 0:
        accuracy = stats['correct_predictions'] / total_steps * 100
        print(f"Overall accuracy: {accuracy:.1f}%")
        print(f"Total prediction steps: {total_steps}")

    print("\nPer-task accuracy:")
    for task_name, task_stats in stats['tasks'].items():
        if task_stats['total_steps'] > 0:
            task_accuracy = task_stats['correct_steps'] / task_stats['total_steps'] * 100
            print(f"  {task_name:30s}: {task_stats['correct_steps']:3d}/{task_stats['total_steps']:3d} steps ({task_accuracy:5.1f}%)")

    # Save statistics (without examples to keep it clean)
    stats_clean = {k: v for k, v in stats.items() if k != 'tasks'}
    stats_clean['tasks'] = {k: {kk: vv for kk, vv in v.items() if kk != 'examples'}
                           for k, v in stats['tasks'].items()}

    stats_file = output_base / "processing_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats_clean, f, indent=2)

    print(f"\nCleaned files saved to: {output_base}/")
    print(f"Folder structure preserved: {list(stats['tasks'].keys())}")
    print(f"Statistics saved to: {stats_file}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        # First argument is always the input model path
        model_path = sys.argv[1]
        # Second argument is optional output path
        output_path = sys.argv[2] if len(sys.argv) >= 3 else None
        process_model_results(model_path, output_path)
    else:
        # Default behavior for backward compatibility
        process_model_results()