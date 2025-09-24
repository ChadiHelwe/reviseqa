#!/bin/bash
# Test script for detailed evaluation with just two examples
# Usage: ./scripts/test_detailed_eval.sh

# Change to project root directory

# Create test directories
mkdir -p models_results
mkdir -p detailed_models_results

echo "Results will be saved to models_results/ and detailed_models_results/"

model_name="google/gemini-2.5-flash"
# Run evaluation with small batch size and single model for testing
python src/evaluation.py \
    --data-dir reviseqa_data/nl/testing_models_data/ \
    --results-dir models_results/ \
    --detailed-output-dir detailed_models_results/$model_name \
    --batch-size 1 \
    --model-name $model_name \
    --guided

echo ""
echo "Test completed! Check the following directories:"
echo "- Summary metrics: models_results/"
echo "- Detailed JSON files: detailed_models_results/$model_name/"
echo ""
