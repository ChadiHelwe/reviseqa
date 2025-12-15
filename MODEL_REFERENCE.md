# Model Reference Guide

Complete reference for Gemma, Qwen, and Llama models up to 15B for LoRA finetuning.

## Quick Selection Guide

### No HF Access Required ✅
**Qwen Models** - Best choice for immediate use:
- **Qwen2.5-7B-Instruct** - Recommended for most users (7B)
- **Qwen2.5-14B-Instruct** - Best performance (14B)
- **Qwen2.5-3B-Instruct** - Fast training (3B)

### Requires HF Access 🔐
**Gemma & Llama** - Need approval from Meta/Google:
- **Gemma-2-9B-IT** - Excellent reasoning (9B)
- **Llama-3.1-8B-Instruct** - Strong baseline (8B)
- **Llama-2-13B-Chat** - Proven performance (13B)

## Complete Model List

### Gemma Models (Google)

| Model | Size | HF Access | Best For |
|-------|------|-----------|----------|
| `google/gemma-2b` | 2B | Required | Quick experiments |
| `google/gemma-2b-it` | 2B | Required | Instruction following |
| `google/gemma-7b` | 7B | Required | General purpose |
| `google/gemma-7b-it` | 7B | Required | Chat/instructions |
| `google/gemma-2-2b` | 2B | Required | Latest small model |
| `google/gemma-2-2b-it` | 2B | Required | Latest small instruct |
| `google/gemma-2-9b` | 9B | Required | Latest medium base |
| `google/gemma-2-9b-it` | 9B | Required | **Best reasoning** |

**Recommended**: `google/gemma-2-9b-it` (9B)

### Qwen Models (Alibaba)

| Model | Size | HF Access | Best For |
|-------|------|-----------|----------|
| `Qwen/Qwen2.5-0.5B` | 0.5B | None | Tiny experiments |
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | None | Tiny instruct |
| `Qwen/Qwen2.5-1.5B` | 1.5B | None | Small experiments |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | None | Small instruct |
| `Qwen/Qwen2.5-3B` | 3B | None | Fast training |
| `Qwen/Qwen2.5-3B-Instruct` | 3B | None | Fast instruct |
| `Qwen/Qwen2.5-7B` | 7B | None | Standard base |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | None | **Recommended 7B** |
| `Qwen/Qwen2.5-14B` | 14B | None | Large base |
| `Qwen/Qwen2.5-14B-Instruct` | 14B | None | **Best overall** |
| `Qwen/Qwen2-7B` | 7B | None | Legacy 7B |
| `Qwen/Qwen2-7B-Instruct` | 7B | None | Legacy 7B instruct |

**Recommended**:
- `Qwen/Qwen2.5-7B-Instruct` (7B) - Best for <10GB VRAM
- `Qwen/Qwen2.5-14B-Instruct` (14B) - Best for 10-16GB VRAM

### Llama Models (Meta)

| Model | Size | HF Access | Best For |
|-------|------|-----------|----------|
| `meta-llama/Llama-3.2-1B` | 1B | Required | Tiny model |
| `meta-llama/Llama-3.2-1B-Instruct` | 1B | Required | Tiny instruct |
| `meta-llama/Llama-3.2-3B` | 3B | Required | Small model |
| `meta-llama/Llama-3.2-3B-Instruct` | 3B | Required | Small instruct |
| `meta-llama/Llama-2-7b-hf` | 7B | Required | Legacy 7B |
| `meta-llama/Llama-2-7b-chat-hf` | 7B | Required | Legacy 7B chat |
| `meta-llama/Meta-Llama-3-8B` | 8B | Required | Llama 3 base |
| `meta-llama/Meta-Llama-3-8B-Instruct` | 8B | Required | Llama 3 instruct |
| `meta-llama/Meta-Llama-3.1-8B` | 8B | Required | Latest 8B base |
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | 8B | Required | **Latest 8B instruct** |
| `meta-llama/Llama-2-13b-hf` | 13B | Required | Legacy 13B |
| `meta-llama/Llama-2-13b-chat-hf` | 13B | Required | **Proven 13B** |

**Recommended**:
- `meta-llama/Meta-Llama-3.1-8B-Instruct` (8B) - Latest, best 8B
- `meta-llama/Llama-2-13b-chat-hf` (13B) - Proven large model

## VRAM Requirements (4-bit Quantization)

| Model Size | Training VRAM | Evaluation VRAM | Batch Size |
|------------|---------------|------------------|------------|
| 0.5-1B | 3-4 GB | 2-3 GB | 8 |
| 1.5-2B | 4-5 GB | 3-4 GB | 8 |
| 3B | 5-6 GB | 4-5 GB | 4 |
| 7-8B | 7-9 GB | 6-8 GB | 4 |
| 9B | 8-10 GB | 7-9 GB | 2-4 |
| 13-14B | 11-14 GB | 10-12 GB | 2 |
| 15B+ | 14-18 GB | 12-16 GB | 1-2 |

**Note**: Without quantization, multiply VRAM by ~4x

## Training Configurations

### Small Models (<=8B) - Consumer GPUs

```bash
# Qwen2.5 7B (Recommended)
python lora_finetune.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --dataset_name folio \
  --output_dir ./lora_qwen2.5-7b_folio \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --use_4bit

# Llama 3.1 8B
python lora_finetune.py \
  --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dataset_name folio \
  --output_dir ./lora_llama3.1-8b_folio \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --use_4bit
```

### Medium Models (9-15B) - Professional GPUs

```bash
# Qwen2.5 14B (Best Performance)
python lora_finetune.py \
  --model_name Qwen/Qwen2.5-14B-Instruct \
  --dataset_name folio \
  --output_dir ./lora_qwen2.5-14b_folio \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --use_4bit

# Gemma 2 9B
python lora_finetune.py \
  --model_name google/gemma-2-9b-it \
  --dataset_name folio \
  --output_dir ./lora_gemma2-9b_folio \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --use_4bit

# Llama 2 13B
python lora_finetune.py \
  --model_name meta-llama/Llama-2-13b-chat-hf \
  --dataset_name folio \
  --output_dir ./lora_llama2-13b_folio \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --use_4bit
```

## Accessing Gated Models

### Llama Models (Meta)
1. Visit: https://huggingface.co/meta-llama
2. Click on model (e.g., Meta-Llama-3.1-8B-Instruct)
3. Click "Request Access" button
4. Wait for approval (usually <1 hour)
5. Create access token: https://huggingface.co/settings/tokens
6. Login: `huggingface-cli login`

### Gemma Models (Google)
1. Visit: https://huggingface.co/google/gemma-2-9b-it
2. Click "Agree and access repository"
3. Accept terms
4. Create access token: https://huggingface.co/settings/tokens
5. Login: `huggingface-cli login`

## Model Selection Decision Tree

```
Do you have HF access to Meta/Google models?
├─ NO → Use Qwen models
│   ├─ <10GB VRAM → Qwen2.5-7B-Instruct ✓
│   └─ 10-16GB VRAM → Qwen2.5-14B-Instruct ✓✓
│
└─ YES → Choose based on size/performance
    ├─ Want best reasoning → Gemma-2-9B-IT
    ├─ Want latest Llama → Llama-3.1-8B-Instruct
    ├─ Want proven large → Llama-2-13B-Chat
    └─ Want best overall → Qwen2.5-14B-Instruct ✓✓
```

## Performance Expectations

### Logical Reasoning (FOLIO/ProofWriter)

Based on general benchmarks:

**Small Models (<=8B)**
- Qwen2.5-7B: ~75-80% accuracy (expected)
- Llama-3.1-8B: ~70-75% accuracy
- Gemma-7B: ~70-75% accuracy

**Medium Models (9-15B)**
- Qwen2.5-14B: ~80-85% accuracy (expected)
- Gemma-2-9B: ~78-82% accuracy
- Llama-2-13B: ~75-80% accuracy

**Note**: Actual performance depends on training data, hyperparameters, and evaluation setup.

## Training Time Estimates

On single RTX 3090/4090 (24GB VRAM):

| Model Size | FOLIO (3 epochs) | ProofWriter-5k (3 epochs) |
|------------|------------------|---------------------------|
| 7-8B | 2-4 hours | 3-6 hours |
| 9B | 3-5 hours | 4-8 hours |
| 13-14B | 4-7 hours | 6-12 hours |

**With multiple GPUs**: Divide time by number of GPUs (approximately)

## Recommended Training Hyperparameters

### All Models (Default)
```bash
--num_epochs 3
--learning_rate 2e-4
--lora_r 16
--lora_alpha 32
--lora_dropout 0.05
--max_seq_length 2048
--use_4bit
```

### For Larger Models (13-15B)
```bash
--num_epochs 3
--learning_rate 1e-4  # Lower LR
--lora_r 32           # Higher rank
--lora_alpha 64
--batch_size 2        # Smaller batch
--gradient_accumulation_steps 8  # More accumulation
```

### For Smaller Models (<=3B)
```bash
--num_epochs 5        # More epochs
--learning_rate 3e-4  # Higher LR
--lora_r 8            # Lower rank
--lora_alpha 16
--batch_size 8        # Larger batch
```

## Using the Scripts

### Train All Recommended Models
```bash
# Edit train_all_models.sh to uncomment desired models
./train_all_models.sh folio
./train_all_models.sh proofwriter
```

### Evaluate All Trained Models
```bash
./evaluate_all_models.sh reviseqa_data/nl/verified-400/
```

### Train Single Model
```bash
python lora_finetune.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --dataset_name folio \
  --output_dir ./lora_qwen2.5-7b_folio
```

### Evaluate Single Model
```bash
python lora_evaluation_complete.py \
  --data-dir reviseqa_data/nl/verified-400/ \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --lora-model ./lora_qwen2.5-7b_folio/final_model \
  --results-dir lora_results/ \
  --detailed-output-dir lora_detailed/
```

## Troubleshooting

### "Model not found" Error
- For Llama/Gemma: Run `huggingface-cli login` and ensure access granted
- Check model name spelling (case-sensitive)
- Verify internet connection

### CUDA Out of Memory
- Reduce `--batch_size` (try 2 or 1)
- Increase `--gradient_accumulation_steps` (try 8 or 16)
- Use `--use_4bit` instead of `--use_8bit`
- Reduce `--max_seq_length` (try 1024)

### Slow Training
- Ensure GPU is being used: `nvidia-smi`
- Check quantization is enabled: `--use_4bit`
- Verify CUDA is properly installed
- Consider using smaller model or fewer epochs

## Summary

### Best Overall Choice
**Qwen2.5-14B-Instruct** - No access restrictions, excellent performance

### Best for Limited VRAM (<10GB)
**Qwen2.5-7B-Instruct** - Great balance of performance and efficiency

### Best Reasoning (with access)
**Gemma-2-9B-IT** - Excellent logical reasoning capabilities

### Best Latest Llama
**Llama-3.1-8B-Instruct** - Most recent Llama model under 15B

All models are production-ready and well-suited for logical reasoning tasks!
