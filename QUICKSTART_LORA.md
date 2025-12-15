# Quick Start Guide: LoRA Finetuning

This guide will help you quickly get started with LoRA finetuning on FOLIO and ProofWriter datasets.

## Step 1: Install Dependencies

```bash
pip install -r requirements_lora.txt
```

**Note**: You may also need to install PyTorch with CUDA support if not already installed:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Step 2: (Optional) Login to Hugging Face

For gated models like Llama, you need to:
1. Get access on Hugging Face
2. Create an access token
3. Login:

```bash
huggingface-cli login
```

## Step 3: Choose Your Model and Dataset

Available models:
- `meta-llama/Llama-2-7b-hf` (requires HF access)
- `meta-llama/Meta-Llama-3-8B` (requires HF access)
- `google/gemma-7b` (requires HF access)
- `google/gemma-2-9b` (requires HF access)
- `Qwen/Qwen2-7B`
- `Qwen/Qwen2.5-7B`
- `mistralai/Mistral-7B-v0.1`

Available datasets:
- `folio` - FOLIO logical reasoning dataset
- `proofwriter` - ProofWriter logical reasoning dataset

## Step 4: Run Finetuning

### Example 1: Finetune Llama-2-7b on FOLIO

```bash
python lora_finetune.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --dataset_name folio \
  --output_dir ./lora_llama2_folio \
  --num_epochs 3 \
  --batch_size 4 \
  --learning_rate 2e-4
```

### Example 2: Finetune Qwen2-7B on ProofWriter

```bash
python lora_finetune.py \
  --model_name Qwen/Qwen2-7B \
  --dataset_name proofwriter \
  --output_dir ./lora_qwen2_proofwriter \
  --num_epochs 3 \
  --batch_size 4 \
  --learning_rate 2e-4
```

### Example 3: Finetune Gemma-7b on FOLIO

```bash
python lora_finetune.py \
  --model_name google/gemma-7b \
  --dataset_name folio \
  --output_dir ./lora_gemma_folio \
  --num_epochs 3 \
  --batch_size 4 \
  --learning_rate 2e-4
```

## Step 5: Run Inference

### Interactive Mode

```bash
python lora_inference.py \
  --base_model meta-llama/Llama-2-7b-hf \
  --lora_model ./lora_llama2_folio/final_model \
  --mode interactive
```

### Single Inference

```bash
python lora_inference.py \
  --base_model meta-llama/Llama-2-7b-hf \
  --lora_model ./lora_llama2_folio/final_model \
  --mode single \
  --instruction "Context: All dogs are animals. Rex is a dog. Question: Is Rex an animal?" \
  --input "The correct option is:"
```

### Batch Inference from File

```bash
python lora_inference.py \
  --base_model meta-llama/Llama-2-7b-hf \
  --lora_model ./lora_llama2_folio/final_model \
  --mode file \
  --input_file Provergen/training_data/folio.json \
  --output_file predictions.json
```

## Step 6: Evaluate Model

```bash
python evaluate_lora_model.py \
  --base_model meta-llama/Llama-2-7b-hf \
  --lora_model ./lora_llama2_folio/final_model \
  --dataset_path Provergen/training_data/folio.json \
  --output_file evaluation_results.json
```

## Step 7: (Optional) Merge LoRA Weights

For easier deployment, you can merge LoRA weights into the base model:

```bash
python merge_lora_weights.py \
  --base_model meta-llama/Llama-2-7b-hf \
  --lora_model ./lora_llama2_folio/final_model \
  --output_path ./merged_llama2_folio
```

## Memory Requirements

Approximate VRAM requirements with 4-bit quantization:

| Model Size | VRAM Required |
|------------|---------------|
| 7B         | 6-8 GB        |
| 8B         | 7-9 GB        |
| 13B        | 10-12 GB      |

If you have limited VRAM, try:
- Reducing `--batch_size` (e.g., to 1 or 2)
- Increasing `--gradient_accumulation_steps` (e.g., to 8 or 16)
- Using `--max_seq_length 1024` instead of default 2048

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size and increase gradient accumulation
python lora_finetune.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --dataset_name folio \
  --output_dir ./lora_llama2_folio \
  --batch_size 1 \
  --gradient_accumulation_steps 16
```

### Can't Access Gated Models

1. Go to the model page on Hugging Face (e.g., https://huggingface.co/meta-llama/Llama-2-7b-hf)
2. Click "Request Access" and wait for approval
3. Create an access token at https://huggingface.co/settings/tokens
4. Run `huggingface-cli login` and enter your token

### Slow Training

- Ensure you're using a GPU: Check with `nvidia-smi`
- Verify CUDA is available in Python:
  ```python
  import torch
  print(torch.cuda.is_available())
  ```

## Advanced Options

### Custom LoRA Configuration

```bash
python lora_finetune.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --dataset_name folio \
  --output_dir ./lora_llama2_folio_custom \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_dropout 0.1 \
  --target_modules q_proj v_proj k_proj o_proj gate_proj up_proj down_proj
```

### Longer Training

```bash
python lora_finetune.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --dataset_name folio \
  --output_dir ./lora_llama2_folio_long \
  --num_epochs 5 \
  --save_steps 50 \
  --logging_steps 5 \
  --warmup_steps 200
```

### Using Full ProofWriter Dataset

```bash
python lora_finetune.py \
  --model_name Qwen/Qwen2-7B \
  --dataset_name proofwriter \
  --dataset_path Provergen/training_data/proofwriter-full.json \
  --output_dir ./lora_qwen2_proofwriter_full \
  --num_epochs 3
```

## Next Steps

1. Experiment with different models and hyperparameters
2. Evaluate on held-out test sets
3. Compare performance across different model architectures
4. Try ensemble methods with multiple finetuned models

For more examples, see [finetune_examples.sh](finetune_examples.sh)

For detailed documentation, see [README_LORA.md](README_LORA.md)
