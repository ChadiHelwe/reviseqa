#!/bin/bash
# Batch evaluation script for multiple models
# Usage: ./scripts/evaluate_all.sh

# Change to project root directory
cd "$(dirname "$0")/../src"

# Create results directory if it doesn't exist
mkdir -p results/newer+2

python evaluation.py --data-dir ../reviseqa_data/nl/verified/ --results-dir results/newer_2/ --guided --batch-size 256 --model-name qwen/qwen3-32b
python evaluation.py --data-dir ../reviseqa_data/nl/verified/ --results-dir results/newer_2/ --guided --batch-size 256 --model-name deepseek/deepseek-chat-v3-0324