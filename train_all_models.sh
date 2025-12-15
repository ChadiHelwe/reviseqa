#!/bin/bash
# Train all recommended models (Gemma, Qwen, Llama up to 15B)
# Usage: ./train_all_models.sh [folio|proofwriter]

set -e  # Exit on error

# Source model configurations
source model_configs.sh

# Default dataset
DATASET="${1:-folio}"
NUM_EPOCHS=3
BATCH_SIZE=4
LEARNING_RATE=2e-4

echo "=============================================================================="
echo "Training All Models"
echo "=============================================================================="
echo "Dataset: $DATASET"
echo "Epochs: $NUM_EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "=============================================================================="
echo ""

# Function to train a model
train_model() {
    local MODEL_NAME=$1
    local MODEL_SHORT_NAME=$2
    local OUTPUT_DIR="./lora_${MODEL_SHORT_NAME}_${DATASET}"

    echo ""
    echo "=========================================="
    echo "Training: $MODEL_SHORT_NAME"
    echo "Model: $MODEL_NAME"
    echo "Output: $OUTPUT_DIR"
    echo "=========================================="

    python lora_finetune.py \
        --model_name "$MODEL_NAME" \
        --dataset_name "$DATASET" \
        --output_dir "$OUTPUT_DIR" \
        --num_epochs $NUM_EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --lora_r 16 \
        --lora_alpha 32 \
        --use_4bit

    if [ $? -eq 0 ]; then
        echo "✓ Successfully trained $MODEL_SHORT_NAME"
    else
        echo "✗ Failed to train $MODEL_SHORT_NAME"
    fi
}

# =============================================================================
# TRAIN SMALL MODELS (<=9B)
# =============================================================================

echo ""
echo "=============================================================================="
echo "PHASE 1: Training Small Models (<=9B)"
echo "=============================================================================="

# Qwen2.5 7B (Recommended - no HF access needed)
train_model "$QWEN2_5_7B_INSTRUCT" "qwen2.5-7b"

# Gemma 2 9B (Requires HF access)
# Uncomment if you have access:
# train_model "$GEMMA2_9B_IT" "gemma2-9b"

# Llama 3.1 8B (Requires HF access)
# Uncomment if you have access:
# train_model "$LLAMA3_1_8B_INSTRUCT" "llama3.1-8b"

# Llama 2 7B (Requires HF access)
# Uncomment if you have access:
# train_model "$LLAMA2_7B_CHAT" "llama2-7b"

# =============================================================================
# TRAIN MEDIUM MODELS (10-15B) - Requires more VRAM
# =============================================================================

echo ""
echo "=============================================================================="
echo "PHASE 2: Training Medium Models (10-15B)"
echo "=============================================================================="

# Qwen2.5 14B (Recommended - no HF access needed)
train_model "$QWEN2_5_14B_INSTRUCT" "qwen2.5-14b"

# Llama 2 13B (Requires HF access)
# Uncomment if you have access:
# train_model "$LLAMA2_13B_CHAT" "llama2-13b"

# =============================================================================
# OPTIONAL: Additional Models
# =============================================================================

# Uncomment to train additional models:

# Qwen2.5 3B (Fast training)
# train_model "$QWEN2_5_3B_INSTRUCT" "qwen2.5-3b"

# Gemma 7B
# train_model "$GEMMA_7B_IT" "gemma-7b"

# Llama 3.2 3B
# train_model "$LLAMA3_2_3B_INSTRUCT" "llama3.2-3b"

# Qwen2 7B (older version)
# train_model "$QWEN2_7B_INSTRUCT" "qwen2-7b"

echo ""
echo "=============================================================================="
echo "Training Complete!"
echo "=============================================================================="
echo "All models trained successfully!"
echo "Output directories: ./lora_*_${DATASET}/"
echo "=============================================================================="
