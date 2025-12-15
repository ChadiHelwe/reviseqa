#!/bin/bash
# Example evaluation scripts for LoRA finetuned models
# These mirror the evaluations done in evaluation.py

# Set your model paths
BASE_MODEL="meta-llama/Llama-2-7b-hf"
LORA_MODEL="./lora_llama2_folio/final_model"

# Example 1: Evaluate on FOLIO dev set with CoT
# python lora_evaluation.py \
#   --base_model $BASE_MODEL \
#   --lora_model $LORA_MODEL \
#   --dataset_name FOLIO \
#   --split dev \
#   --mode CoT \
#   --trained_model \
#   --output_dir lora_results/

# Example 2: Evaluate on FOLIO dev set with Direct (no reasoning)
# python lora_evaluation.py \
#   --base_model $BASE_MODEL \
#   --lora_model $LORA_MODEL \
#   --dataset_name FOLIO \
#   --split dev \
#   --mode Direct \
#   --trained_model \
#   --output_dir lora_results/

# Example 3: Evaluate on ProofWriter test set with CoT
# python lora_evaluation.py \
#   --base_model $BASE_MODEL \
#   --lora_model $LORA_MODEL \
#   --dataset_name ProofWriter \
#   --split test \
#   --mode CoT \
#   --trained_model \
#   --output_dir lora_results/

# Example 4: Evaluate on ProofWriter test set with Direct
# python lora_evaluation.py \
#   --base_model $BASE_MODEL \
#   --lora_model $LORA_MODEL \
#   --dataset_name ProofWriter \
#   --split test \
#   --mode Direct \
#   --trained_model \
#   --output_dir lora_results/

# Example 5: Evaluate on ProntoQA dev set
# python lora_evaluation.py \
#   --base_model $BASE_MODEL \
#   --lora_model $LORA_MODEL \
#   --dataset_name ProntoQA \
#   --split dev \
#   --mode CoT \
#   --trained_model \
#   --output_dir lora_results/

# Example 6: Evaluate with specific range (first 100 examples)
# python lora_evaluation.py \
#   --base_model $BASE_MODEL \
#   --lora_model $LORA_MODEL \
#   --dataset_name FOLIO \
#   --split dev \
#   --mode CoT \
#   --trained_model \
#   --start 0 \
#   --end 100 \
#   --output_dir lora_results/

# Example 7: Evaluate with higher temperature (more creative)
# python lora_evaluation.py \
#   --base_model $BASE_MODEL \
#   --lora_model $LORA_MODEL \
#   --dataset_name FOLIO \
#   --split dev \
#   --mode CoT \
#   --trained_model \
#   --temperature 0.7 \
#   --output_dir lora_results/

# Example 8: Evaluate with verbose output
# python lora_evaluation.py \
#   --base_model $BASE_MODEL \
#   --lora_model $LORA_MODEL \
#   --dataset_name FOLIO \
#   --split dev \
#   --mode CoT \
#   --trained_model \
#   --verbose \
#   --start 0 \
#   --end 5 \
#   --output_dir lora_results/

# Example 9: Evaluate Qwen2 model
# python lora_evaluation.py \
#   --base_model Qwen/Qwen2-7B \
#   --lora_model ./lora_qwen2_folio/final_model \
#   --dataset_name FOLIO \
#   --split dev \
#   --mode CoT \
#   --trained_model \
#   --output_dir lora_results/

# Example 10: Evaluate Gemma model
# python lora_evaluation.py \
#   --base_model google/gemma-7b \
#   --lora_model ./lora_gemma_proofwriter/final_model \
#   --dataset_name ProofWriter \
#   --split test \
#   --mode CoT \
#   --trained_model \
#   --output_dir lora_results/

# Example 11: Comprehensive evaluation on all datasets
# for DATASET in FOLIO ProofWriter ProntoQA; do
#   for MODE in CoT Direct; do
#     python lora_evaluation.py \
#       --base_model $BASE_MODEL \
#       --lora_model $LORA_MODEL \
#       --dataset_name $DATASET \
#       --split dev \
#       --mode $MODE \
#       --trained_model \
#       --output_dir lora_results/
#   done
# done

# Example 12: Batch evaluation script for multiple models
# MODELS=(
#   "meta-llama/Llama-2-7b-hf:./lora_llama2_folio/final_model"
#   "Qwen/Qwen2-7B:./lora_qwen2_folio/final_model"
#   "google/gemma-7b:./lora_gemma_folio/final_model"
# )
#
# for MODEL_PAIR in "${MODELS[@]}"; do
#   IFS=':' read -r BASE LORA <<< "$MODEL_PAIR"
#   python lora_evaluation.py \
#     --base_model "$BASE" \
#     --lora_model "$LORA" \
#     --dataset_name FOLIO \
#     --split dev \
#     --mode CoT \
#     --trained_model \
#     --output_dir lora_results/
# done

echo "Evaluation examples provided. Uncomment and run the desired example."
