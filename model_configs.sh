#!/bin/bash
# Model Configurations for Training and Evaluation
# Includes Gemma, Qwen, and Llama models up to 15B

# =============================================================================
# GEMMA MODELS
# =============================================================================

# Gemma 2B
GEMMA_2B="google/gemma-2b"
GEMMA_2B_IT="google/gemma-2b-it"  # Instruction-tuned

# Gemma 7B
GEMMA_7B="google/gemma-7b"
GEMMA_7B_IT="google/gemma-7b-it"

# Gemma 2 (2B)
GEMMA2_2B="google/gemma-2-2b"
GEMMA2_2B_IT="google/gemma-2-2b-it"

# Gemma 2 (9B)
GEMMA2_9B="google/gemma-2-9b"
GEMMA2_9B_IT="google/gemma-2-9b-it"

# =============================================================================
# QWEN MODELS
# =============================================================================

# Qwen2 Series
QWEN2_0_5B="Qwen/Qwen2-0.5B"
QWEN2_0_5B_INSTRUCT="Qwen/Qwen2-0.5B-Instruct"

QWEN2_1_5B="Qwen/Qwen2-1.5B"
QWEN2_1_5B_INSTRUCT="Qwen/Qwen2-1.5B-Instruct"

QWEN2_7B="Qwen/Qwen2-7B"
QWEN2_7B_INSTRUCT="Qwen/Qwen2-7B-Instruct"

# Qwen2.5 Series
QWEN2_5_0_5B="Qwen/Qwen2.5-0.5B"
QWEN2_5_0_5B_INSTRUCT="Qwen/Qwen2.5-0.5B-Instruct"

QWEN2_5_1_5B="Qwen/Qwen2.5-1.5B"
QWEN2_5_1_5B_INSTRUCT="Qwen/Qwen2.5-1.5B-Instruct"

QWEN2_5_3B="Qwen/Qwen2.5-3B"
QWEN2_5_3B_INSTRUCT="Qwen/Qwen2.5-3B-Instruct"

QWEN2_5_7B="Qwen/Qwen2.5-7B"
QWEN2_5_7B_INSTRUCT="Qwen/Qwen2.5-7B-Instruct"

QWEN2_5_14B="Qwen/Qwen2.5-14B"
QWEN2_5_14B_INSTRUCT="Qwen/Qwen2.5-14B-Instruct"

# =============================================================================
# LLAMA MODELS
# =============================================================================

# Llama 2
LLAMA2_7B="meta-llama/Llama-2-7b-hf"
LLAMA2_7B_CHAT="meta-llama/Llama-2-7b-chat-hf"

LLAMA2_13B="meta-llama/Llama-2-13b-hf"
LLAMA2_13B_CHAT="meta-llama/Llama-2-13b-chat-hf"

# Llama 3
LLAMA3_8B="meta-llama/Meta-Llama-3-8B"
LLAMA3_8B_INSTRUCT="meta-llama/Meta-Llama-3-8B-Instruct"

# Llama 3.1
LLAMA3_1_8B="meta-llama/Meta-Llama-3.1-8B"
LLAMA3_1_8B_INSTRUCT="meta-llama/Meta-Llama-3.1-8B-Instruct"

# Llama 3.2
LLAMA3_2_1B="meta-llama/Llama-3.2-1B"
LLAMA3_2_1B_INSTRUCT="meta-llama/Llama-3.2-1B-Instruct"

LLAMA3_2_3B="meta-llama/Llama-3.2-3B"
LLAMA3_2_3B_INSTRUCT="meta-llama/Llama-3.2-3B-Instruct"

# Llama 3.3
LLAMA3_3_70B="meta-llama/Llama-3.3-70B-Instruct"  # Note: >15B but included for reference

# =============================================================================
# RECOMMENDED MODELS (Up to 15B)
# =============================================================================

# Best for training on consumer hardware (<=8B)
RECOMMENDED_SMALL=(
    "$QWEN2_5_7B_INSTRUCT"      # Qwen2.5 7B - Great performance, no access restrictions
    "$GEMMA2_9B_IT"             # Gemma 2 9B - Strong reasoning, requires HF access
    "$LLAMA3_1_8B_INSTRUCT"     # Llama 3.1 8B - Good baseline, requires HF access
)

# Medium models (8-15B) - Requires more VRAM
RECOMMENDED_MEDIUM=(
    "$QWEN2_5_14B_INSTRUCT"     # Qwen2.5 14B - Best in class, no restrictions
    "$LLAMA2_13B_CHAT"          # Llama 2 13B - Proven performance, requires HF access
)

# All base models (non-instruct) for finetuning
BASE_MODELS_SMALL=(
    "$QWEN2_5_7B"
    "$GEMMA2_9B"
    "$LLAMA3_1_8B"
)

BASE_MODELS_MEDIUM=(
    "$QWEN2_5_14B"
    "$LLAMA2_13B"
)

# =============================================================================
# HARDWARE REQUIREMENTS (Approximate VRAM with 4-bit quantization)
# =============================================================================

# 1-3B models:  4-6 GB VRAM
# 7-8B models:  6-8 GB VRAM
# 9B models:    7-9 GB VRAM
# 13-14B models: 10-12 GB VRAM
# 15B+ models:  12-16 GB VRAM

# =============================================================================
# MODEL SELECTION GUIDE
# =============================================================================

# For NO ACCESS RESTRICTIONS (no HF approval needed):
# - Qwen2.5 series (0.5B to 14B)
# - Best choice: Qwen2.5-7B-Instruct or Qwen2.5-14B-Instruct

# For BEST PERFORMANCE (requires HF access):
# - Gemma 2 9B (excellent reasoning)
# - Llama 3.1 8B (strong baseline)
# - Llama 2 13B (proven performance)

# For LIMITED VRAM (<10GB):
# - Qwen2.5-7B-Instruct (recommended)
# - Gemma-2B-IT (fast training)
# - Llama-3.2-3B-Instruct

# For MAXIMUM PERFORMANCE (10-16GB VRAM):
# - Qwen2.5-14B-Instruct (best choice)
# - Llama-2-13B-chat
# - Gemma-2-9B-IT

# =============================================================================
# GPU DEVICE SELECTION
# =============================================================================

# Set which GPU to use (default: GPU 0)
# Override by setting environment variable: export GPU_DEVICE=1
GPU_DEVICE="${GPU_DEVICE:-0}"

# For multi-GPU training, specify multiple devices: GPU_DEVICE="0,1"
# For CPU only (slow): GPU_DEVICE=""

# Export CUDA_VISIBLE_DEVICES to control which GPU(s) are used
export CUDA_VISIBLE_DEVICES="$GPU_DEVICE"

echo "Model configurations loaded!"
echo ""
echo "GPU Configuration:"
echo "  - Using GPU(s): ${GPU_DEVICE:-CPU only}"
echo "  - Override with: export GPU_DEVICE=1 (before running scripts)"
echo ""
echo "Available model families:"
echo "  - Gemma: 2B, 7B, 9B models"
echo "  - Qwen: 0.5B to 14B models (no HF access required)"
echo "  - Llama: 1B to 13B models (HF access required)"
echo ""
echo "Recommended for training:"
echo "  - Small (<=8GB VRAM): $QWEN2_5_7B_INSTRUCT"
echo "  - Medium (10-16GB VRAM): $QWEN2_5_14B_INSTRUCT"
echo ""
