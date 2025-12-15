#!/bin/bash
# Run LoRA model evaluation matching src/evaluation.py methodology
# Usage: ./run_lora_evaluation.sh [gpu_device]
# Examples:
#   ./run_lora_evaluation.sh       # Use GPU 0 (default)
#   ./run_lora_evaluation.sh 1     # Use GPU 1
#   ./run_lora_evaluation.sh "0,1" # Use GPUs 0 and 1
#   GPU_DEVICE=2 ./run_lora_evaluation.sh  # Use GPU 2

# Set GPU device if provided
if [ -n "$1" ]; then
    export CUDA_VISIBLE_DEVICES="$1"
    echo "Using GPU(s): $1"
else
    export CUDA_VISIBLE_DEVICES="${GPU_DEVICE:-0}"
    echo "Using GPU(s): ${CUDA_VISIBLE_DEVICES}"
fi

# Configuration
DATA_DIR="reviseqa_data/nl/verified-400/"
BASE_MODEL="Qwen/Qwen2-7B"
LORA_MODEL="./lora_qwen2_folio/final_model"
RESULTS_DIR="lora_models_results/"
DETAILED_OUTPUT_DIR="lora_detailed_models_results/"
BATCH_SIZE=4

# Create output directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$DETAILED_OUTPUT_DIR"

echo "=============================================================================="
echo "LoRA Model Evaluation"
echo "=============================================================================="
echo "Data Directory: $DATA_DIR"
echo "Base Model: $BASE_MODEL"
echo "LoRA Model: $LORA_MODEL"
echo "Results Directory: $RESULTS_DIR"
echo "Detailed Output: $DETAILED_OUTPUT_DIR"
echo "Batch Size: $BATCH_SIZE"
echo "=============================================================================="
echo ""

# Run evaluation
python lora_evaluation_complete.py \
    --data-dir "$DATA_DIR" \
    --base-model "$BASE_MODEL" \
    --lora-model "$LORA_MODEL" \
    --results-dir "$RESULTS_DIR" \
    --detailed-output-dir "$DETAILED_OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --use-4bit

echo ""
echo "=============================================================================="
echo "Evaluation Complete!"
echo "=============================================================================="
echo "Summary metrics: $RESULTS_DIR"
echo "Detailed JSON files: $DETAILED_OUTPUT_DIR"
echo "=============================================================================="
