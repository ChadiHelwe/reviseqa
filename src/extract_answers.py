#!/usr/bin/env python3
"""
Extract answers from prediction fields in model results
"""

import json
import re
import os
from pathlib import Path


def extract_answer_from_prediction(prediction):
    """
    Extract the answer from a prediction string that may contain JSON with reasoning and answer.

    Args:
        prediction: String containing the prediction, possibly with JSON structure

    Returns:
        The extracted answer value, or the original prediction if parsing fails
    """
    if not isinstance(prediction, str):
        return prediction

    # Try to find JSON-like content within the prediction
    # Look for patterns like {"reasoning": "...", "answer": ...}

    # First, try to extract JSON block from the prediction
    json_match = re.search(r'\{[^}]*"answer"[^}]*\}', prediction, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            parsed = json.loads(json_str)
            if 'answer' in parsed:
                return parsed['answer']
        except json.JSONDecodeError:
            pass

    # If no JSON block found, look for answer field directly
    answer_match = re.search(r'"answer":\s*(true|false|True|False|\d+|"[^"]*")', prediction, re.IGNORECASE)
    if answer_match:
        answer_str = answer_match.group(1)
        # Clean up the answer
        if answer_str.lower() in ['true', 'false']:
            return answer_str.lower() == 'true'
        elif answer_str.startswith('"') and answer_str.endswith('"'):
            return answer_str[1:-1]  # Remove quotes
        elif answer_str.isdigit():
            return int(answer_str)
        else:
            return answer_str

    # If no structured answer found, return original prediction
    return prediction


def process_model_results_file(file_path):
    """
    Process a single model results JSON file and extract answers.

    Args:
        file_path: Path to the JSON file

    Returns:
        Dictionary with extracted answers
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        if 'prediction' in data:
            original_prediction = data['prediction']
            extracted_answer = extract_answer_from_prediction(original_prediction)

            return {
                'file': str(file_path),
                'original_prediction': original_prediction,
                'extracted_answer': extracted_answer,
                'answer_type': type(extracted_answer).__name__
            }
    except Exception as e:
        return {
            'file': str(file_path),
            'error': str(e)
        }

    return None


def process_model_directory(model_dir):
    """
    Process all result files for a specific model and extract answers.

    Args:
        model_dir: Path to model directory

    Returns:
        List of processed results
    """
    results = []
    model_path = Path(model_dir)

    # Find all JSON files in the model directory
    for json_file in model_path.rglob('*.json'):
        result = process_model_results_file(json_file)
        if result:
            results.append(result)

    return results


def process_all_models(results_dir='detailed_models_results'):
    """
    Process all model results and extract answers.

    Args:
        results_dir: Directory containing model results

    Returns:
        Dictionary with results for each model
    """
    results_path = Path(results_dir)
    all_results = {}

    # Iterate through provider directories
    for provider_dir in results_path.iterdir():
        if provider_dir.is_dir():
            provider_name = provider_dir.name
            all_results[provider_name] = {}

            # Iterate through model directories
            for model_dir in provider_dir.iterdir():
                if model_dir.is_dir():
                    model_name = model_dir.name
                    print(f"Processing {provider_name}/{model_name}...")

                    model_results = process_model_directory(model_dir)
                    all_results[provider_name][model_name] = model_results

    return all_results


def save_extracted_answers(results, output_file='extracted_answers.json'):
    """
    Save extracted answers to a JSON file.

    Args:
        results: Results from process_all_models
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Extracted answers saved to {output_file}")


def create_clean_predictions_file(results, output_file='clean_predictions.json'):
    """
    Create a clean file with just the extracted answers for each result.

    Args:
        results: Results from process_all_models
        output_file: Output file path
    """
    clean_data = {}

    for provider, models in results.items():
        clean_data[provider] = {}
        for model, model_results in models.items():
            clean_data[provider][model] = {}

            for result in model_results:
                if 'extracted_answer' in result and 'file' in result:
                    # Use relative file path as key
                    file_key = result['file'].replace('detailed_models_results/', '')
                    clean_data[provider][model][file_key] = result['extracted_answer']

    with open(output_file, 'w') as f:
        json.dump(clean_data, f, indent=2, default=str)

    print(f"Clean predictions saved to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract answers from model prediction results")
    parser.add_argument("--input-dir", default="detailed_models_results", help="Input directory with model results")
    parser.add_argument("--output", default="extracted_answers.json", help="Output file for full results")
    parser.add_argument("--clean-output", default="clean_predictions.json", help="Output file for clean predictions only")
    parser.add_argument("--sample", type=int, help="Process only a sample of files for testing")

    args = parser.parse_args()

    print("Processing model results to extract answers...")
    results = process_all_models(args.input_dir)

    print(f"Processed {sum(len(models) for models in results.values())} models")

    # Save full results
    save_extracted_answers(results, args.output)

    # Save clean predictions
    create_clean_predictions_file(results, args.clean_output)

    # Print summary
    total_files = sum(len(model_results) for provider in results.values()
                     for model_results in provider.values())
    print(f"Total files processed: {total_files}")