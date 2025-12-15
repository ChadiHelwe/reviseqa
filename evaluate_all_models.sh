#!/bin/bash
# Evaluate all trained models on reviseqa_data
# Usage: ./evaluate_all_models.sh [data_dir] [gpu_device]
# Examples:
#   ./evaluate_all_models.sh reviseqa_data/nl/verified-400/          # Use GPU 0
#   ./evaluate_all_models.sh reviseqa_data/nl/verified-400/ 1        # Use GPU 1
#   ./evaluate_all_models.sh reviseqa_data/nl/verified-400/ "0,1"    # Use GPUs 0,1
#   GPU_DEVICE=2 ./evaluate_all_models.sh reviseqa_data/nl/verified-400/  # Use GPU 2

set -e  # Exit on error

# Set GPU device if provided as second argument
if [ -n "$2" ]; then
    export GPU_DEVICE="$2"
fi

# Source model configurations (this will set CUDA_VISIBLE_DEVICES)
source model_configs.sh

# Configuration
DATA_DIR="${1:-reviseqa_data/nl/verified-400/}"
RESULTS_DIR="lora_models_results"
DETAILED_OUTPUT_DIR="lora_detailed_models_results"
BATCH_SIZE=4

# Create output directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$DETAILED_OUTPUT_DIR"

echo "=============================================================================="
echo "Evaluating All Trained Models"
echo "=============================================================================="
echo "Data Directory: $DATA_DIR"
echo "Results Directory: $RESULTS_DIR"
echo "Detailed Output: $DETAILED_OUTPUT_DIR"
echo "Batch Size: $BATCH_SIZE"
echo "=============================================================================="
echo ""

# Function to evaluate a model
evaluate_model() {
    local BASE_MODEL=$1
    local LORA_PATH=$2
    local MODEL_NAME=$3

    echo ""
    echo "=========================================="
    echo "Evaluating: $MODEL_NAME"
    echo "Base Model: $BASE_MODEL"
    echo "LoRA Path: $LORA_PATH"
    echo "=========================================="

    if [ ! -d "$LORA_PATH" ]; then
        echo "⚠ Skipping $MODEL_NAME - LoRA model not found at $LORA_PATH"
        return
    fi

    python lora_evaluation_complete.py \
        --data-dir "$DATA_DIR" \
        --base-model "$BASE_MODEL" \
        --lora-model "$LORA_PATH" \
        --results-dir "$RESULTS_DIR" \
        --detailed-output-dir "$DETAILED_OUTPUT_DIR/$MODEL_NAME" \
        --batch-size $BATCH_SIZE \
        --use-4bit

    if [ $? -eq 0 ]; then
        echo "✓ Successfully evaluated $MODEL_NAME"
    else
        echo "✗ Failed to evaluate $MODEL_NAME"
    fi
}

# =============================================================================
# EVALUATE SMALL MODELS (<=9B)
# =============================================================================

echo ""
echo "=============================================================================="
echo "PHASE 1: Evaluating Small Models (<=9B)"
echo "=============================================================================="

# Qwen2.5 7B on FOLIO
evaluate_model "$QWEN2_5_7B_INSTRUCT" "./lora_qwen2.5-7b_folio/final_model" "qwen2.5-7b-folio"

# Qwen2.5 7B on ProofWriter
evaluate_model "$QWEN2_5_7B_INSTRUCT" "./lora_qwen2.5-7b_proofwriter/final_model" "qwen2.5-7b-proofwriter"

# Gemma 2 9B on FOLIO
evaluate_model "$GEMMA2_9B_IT" "./lora_gemma2-9b_folio/final_model" "gemma2-9b-folio"

# Gemma 2 9B on ProofWriter
evaluate_model "$GEMMA2_9B_IT" "./lora_gemma2-9b_proofwriter/final_model" "gemma2-9b-proofwriter"

# Llama 3.1 8B on FOLIO
evaluate_model "$LLAMA3_1_8B_INSTRUCT" "./lora_llama3.1-8b_folio/final_model" "llama3.1-8b-folio"

# Llama 3.1 8B on ProofWriter
evaluate_model "$LLAMA3_1_8B_INSTRUCT" "./lora_llama3.1-8b_proofwriter/final_model" "llama3.1-8b-proofwriter"

# Llama 2 7B on FOLIO
evaluate_model "$LLAMA2_7B_CHAT" "./lora_llama2-7b_folio/final_model" "llama2-7b-folio"

# Llama 2 7B on ProofWriter
evaluate_model "$LLAMA2_7B_CHAT" "./lora_llama2-7b_proofwriter/final_model" "llama2-7b-proofwriter"

# =============================================================================
# EVALUATE MEDIUM MODELS (10-15B)
# =============================================================================

echo ""
echo "=============================================================================="
echo "PHASE 2: Evaluating Medium Models (10-15B)"
echo "=============================================================================="

# Qwen2.5 14B on FOLIO
evaluate_model "$QWEN2_5_14B_INSTRUCT" "./lora_qwen2.5-14b_folio/final_model" "qwen2.5-14b-folio"

# Qwen2.5 14B on ProofWriter
evaluate_model "$QWEN2_5_14B_INSTRUCT" "./lora_qwen2.5-14b_proofwriter/final_model" "qwen2.5-14b-proofwriter"

# Llama 2 13B on FOLIO
evaluate_model "$LLAMA2_13B_CHAT" "./lora_llama2-13b_folio/final_model" "llama2-13b-folio"

# Llama 2 13B on ProofWriter
evaluate_model "$LLAMA2_13B_CHAT" "./lora_llama2-13b_proofwriter/final_model" "llama2-13b-proofwriter"

# =============================================================================
# OPTIONAL: Additional Models
# =============================================================================

# Uncomment to evaluate additional models:

# Qwen2.5 3B
# evaluate_model "$QWEN2_5_3B_INSTRUCT" "./lora_qwen2.5-3b_folio/final_model" "qwen2.5-3b-folio"

# Gemma 7B
# evaluate_model "$GEMMA_7B_IT" "./lora_gemma-7b_folio/final_model" "gemma-7b-folio"

# Llama 3.2 3B
# evaluate_model "$LLAMA3_2_3B_INSTRUCT" "./lora_llama3.2-3b_folio/final_model" "llama3.2-3b-folio"

echo ""
echo "=============================================================================="
echo "Evaluation Complete!"
echo "=============================================================================="
echo "Results saved in: $RESULTS_DIR"
echo "Detailed outputs in: $DETAILED_OUTPUT_DIR"
echo ""
echo "Summary of evaluated models:"
ls -1 "$DETAILED_OUTPUT_DIR" 2>/dev/null || echo "No models evaluated yet"
echo "=============================================================================="
