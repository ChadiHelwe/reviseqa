#!/bin/bash
# Example scripts for LoRA finetuning different models

# Example 1: Finetune Llama-2-7b on FOLIO dataset
# python lora_finetune.py \
#   --model_name meta-llama/Llama-2-7b-hf \
#   --dataset_name folio \
#   --output_dir ./lora_llama2_folio \
#   --num_epochs 3 \
#   --batch_size 4 \
#   --gradient_accumulation_steps 4 \
#   --learning_rate 2e-4 \
#   --lora_r 16 \
#   --lora_alpha 32

# Example 2: Finetune Llama-3-8b on ProofWriter dataset
# python lora_finetune.py \
#   --model_name meta-llama/Meta-Llama-3-8B \
#   --dataset_name proofwriter \
#   --output_dir ./lora_llama3_proofwriter \
#   --num_epochs 3 \
#   --batch_size 4 \
#   --learning_rate 2e-4

# Example 3: Finetune Gemma-7b on FOLIO dataset
# python lora_finetune.py \
#   --model_name google/gemma-7b \
#   --dataset_name folio \
#   --output_dir ./lora_gemma_folio \
#   --num_epochs 3 \
#   --batch_size 4 \
#   --learning_rate 2e-4

# Example 4: Finetune Gemma-2-9b on ProofWriter dataset
# python lora_finetune.py \
#   --model_name google/gemma-2-9b \
#   --dataset_name proofwriter \
#   --output_dir ./lora_gemma2_proofwriter \
#   --num_epochs 3 \
#   --batch_size 2 \
#   --gradient_accumulation_steps 8

# Example 5: Finetune Qwen2-7B on FOLIO dataset
# python lora_finetune.py \
#   --model_name Qwen/Qwen2-7B \
#   --dataset_name folio \
#   --output_dir ./lora_qwen2_folio \
#   --num_epochs 3 \
#   --batch_size 4 \
#   --learning_rate 2e-4

# Example 6: Finetune Qwen2.5-7B on ProofWriter dataset (full dataset)
# python lora_finetune.py \
#   --model_name Qwen/Qwen2.5-7B \
#   --dataset_name proofwriter \
#   --dataset_path Provergen/training_data/proofwriter-full.json \
#   --output_dir ./lora_qwen25_proofwriter_full \
#   --num_epochs 3 \
#   --batch_size 4 \
#   --learning_rate 2e-4

# Example 7: Finetune Mistral-7B on FOLIO dataset
# python lora_finetune.py \
#   --model_name mistralai/Mistral-7B-v0.1 \
#   --dataset_name folio \
#   --output_dir ./lora_mistral_folio \
#   --num_epochs 3 \
#   --batch_size 4 \
#   --learning_rate 2e-4

# Example 8: Finetune with 8-bit quantization instead of 4-bit
# python lora_finetune.py \
#   --model_name meta-llama/Llama-2-7b-hf \
#   --dataset_name folio \
#   --use_8bit \
#   --output_dir ./lora_llama2_folio_8bit \
#   --num_epochs 3

# Example 9: Finetune without quantization (requires more VRAM)
# python lora_finetune.py \
#   --model_name meta-llama/Llama-2-7b-hf \
#   --dataset_name folio \
#   --no_quantization \
#   --output_dir ./lora_llama2_folio_no_quant \
#   --num_epochs 3 \
#   --batch_size 1 \
#   --gradient_accumulation_steps 16

# Example 10: Custom LoRA parameters for larger rank
# python lora_finetune.py \
#   --model_name meta-llama/Llama-2-7b-hf \
#   --dataset_name folio \
#   --output_dir ./lora_llama2_folio_r64 \
#   --lora_r 64 \
#   --lora_alpha 128 \
#   --lora_dropout 0.1 \
#   --num_epochs 3

# Example 11: Custom target modules
# python lora_finetune.py \
#   --model_name meta-llama/Llama-2-7b-hf \
#   --dataset_name folio \
#   --output_dir ./lora_llama2_folio_custom \
#   --target_modules q_proj v_proj k_proj o_proj \
#   --num_epochs 3

# Example 12: Longer training with more steps
# python lora_finetune.py \
#   --model_name meta-llama/Llama-2-7b-hf \
#   --dataset_name folio \
#   --output_dir ./lora_llama2_folio_long \
#   --num_epochs 5 \
#   --batch_size 4 \
#   --save_steps 50 \
#   --logging_steps 5 \
#   --warmup_steps 200

echo "Examples provided. Uncomment and run the desired example."
