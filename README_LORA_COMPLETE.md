# Complete LoRA Finetuning and Evaluation Pipeline

This is a complete end-to-end pipeline for LoRA finetuning and evaluation of language models on logical reasoning tasks (FOLIO and ProofWriter datasets).

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Complete Workflow](#complete-workflow)
5. [Files Reference](#files-reference)
6. [Detailed Guides](#detailed-guides)

## Overview

This pipeline enables you to:

1. **Train**: LoRA finetune models (Llama, Gemma, Qwen, etc.) on FOLIO/ProofWriter
2. **Evaluate**: Run the same evaluations as the original `evaluation.py`
3. **Compare**: Compare finetuned models with baseline performance
4. **Deploy**: Merge LoRA weights for production use

### Key Features

✅ Multi-model support (Llama, Gemma, Qwen, Mistral)
✅ Efficient 4-bit/8-bit quantization
✅ Multiple datasets (FOLIO, ProofWriter, ProntoQA)
✅ Both CoT and Direct evaluation modes
✅ Compatible with existing evaluation pipeline
✅ Comprehensive documentation and examples

## Installation

```bash
# Install dependencies
pip install -r requirements_lora.txt

# For gated models (Llama, Gemma), login to Hugging Face
huggingface-cli login

# Verify PyTorch CUDA support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Quick Start

### 1. Train a Model (5 minutes to start)

```bash
# Finetune Qwen2-7B on FOLIO dataset
python lora_finetune.py \
  --model_name Qwen/Qwen2-7B \
  --dataset_name folio \
  --output_dir ./lora_qwen2_folio \
  --num_epochs 3 \
  --batch_size 4
```

### 2. Evaluate the Model

```bash
# Evaluate on FOLIO dev set with CoT
python lora_evaluation.py \
  --base_model Qwen/Qwen2-7B \
  --lora_model ./lora_qwen2_folio/final_model \
  --dataset_name FOLIO \
  --split dev \
  --mode CoT \
  --trained_model \
  --output_dir lora_results/
```

### 3. Check Results

Results are saved in `lora_results/` with accuracy printed to console:
```
Accuracy: 0.8523 (189/222)
```

## Complete Workflow

### Step 1: Choose Your Model

Popular choices:
- **Qwen/Qwen2-7B** - Good performance, no access restrictions
- **meta-llama/Llama-2-7b-hf** - Requires HF access approval
- **google/gemma-7b** - Requires HF access approval
- **mistralai/Mistral-7B-v0.1** - Open access

### Step 2: Prepare Data

Datasets are already prepared in `Provergen/`:
- **Training**: `Provergen/training_data/folio.json`, `proofwriter-5000.json`
- **Evaluation**: `Provergen/logic_data/FOLIO/`, `ProofWriter/`, etc.

### Step 3: Finetune

```bash
# Example: Llama-2 on FOLIO
python lora_finetune.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --dataset_name folio \
  --output_dir ./lora_llama2_folio \
  --num_epochs 3 \
  --batch_size 4 \
  --learning_rate 2e-4

# Example: Gemma on ProofWriter
python lora_finetune.py \
  --model_name google/gemma-7b \
  --dataset_name proofwriter \
  --output_dir ./lora_gemma_proofwriter \
  --num_epochs 3 \
  --batch_size 4
```

Training time (approximate):
- 7B model on FOLIO: 2-4 hours (single GPU)
- 7B model on ProofWriter: 3-6 hours (single GPU)

### Step 4: Evaluate on All Datasets

```bash
# Batch evaluation on all datasets and modes
python batch_evaluate_lora.py \
  --base_model meta-llama/Llama-2-7b-hf \
  --lora_model ./lora_llama2_folio/final_model \
  --datasets FOLIO ProofWriter ProntoQA \
  --splits dev test \
  --modes CoT Direct \
  --output_dir lora_results/
```

This runs 12 evaluations (3 datasets × 2 splits × 2 modes) automatically.

### Step 5: Compare with Baseline

Run baseline evaluation using original script:
```bash
python Provergen/evaluation.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --dataset_name FOLIO \
  --split dev \
  --mode CoT \
  --output_dir baseline_results/
```

Compare results:
- Baseline: `baseline_results/CoT_FOLIO_dev_*.json`
- LoRA: `lora_results/CoT_FOLIO_dev_*.json`

### Step 6: Interactive Testing

```bash
# Test your model interactively
python lora_inference.py \
  --base_model meta-llama/Llama-2-7b-hf \
  --lora_model ./lora_llama2_folio/final_model \
  --mode interactive
```

### Step 7: (Optional) Merge for Deployment

```bash
# Merge LoRA weights into base model
python merge_lora_weights.py \
  --base_model meta-llama/Llama-2-7b-hf \
  --lora_model ./lora_llama2_folio/final_model \
  --output_path ./merged_llama2_folio
```

## Files Reference

### Training Scripts
- **`lora_finetune.py`** - Main training script
- **`finetune_examples.sh`** - Training examples
- **`requirements_lora.txt`** - Python dependencies

### Evaluation Scripts
- **`lora_evaluation.py`** - Main evaluation script (compatible with evaluation.py)
- **`batch_evaluate_lora.py`** - Batch evaluation runner
- **`lora_evaluation_examples.sh`** - Evaluation examples
- **`evaluate_lora_model.py`** - Standalone evaluation with metrics

### Inference Scripts
- **`lora_inference.py`** - Interactive and batch inference
- **`merge_lora_weights.py`** - Merge LoRA into base model

### Documentation
- **`README_LORA.md`** - Training documentation
- **`README_LORA_EVALUATION.md`** - Evaluation documentation
- **`QUICKSTART_LORA.md`** - Quick start guide
- **`README_LORA_COMPLETE.md`** - This file (complete guide)

### Data
- **Training data**: `Provergen/training_data/`
  - `folio.json` - FOLIO training data
  - `proofwriter-5000.json` - ProofWriter subset
  - `proofwriter-full.json` - Full ProofWriter dataset

- **Evaluation data**: `Provergen/logic_data/`
  - `FOLIO/` - dev, train splits
  - `ProofWriter/` - train, dev, test splits
  - `ProntoQA/` - dev split
  - `icl_examples/` - Few-shot examples

## Detailed Guides

### Training Guide

See [README_LORA.md](README_LORA.md) for:
- Detailed command-line arguments
- Hyperparameter tuning
- Memory optimization
- LoRA configuration
- Troubleshooting

### Evaluation Guide

See [README_LORA_EVALUATION.md](README_LORA_EVALUATION.md) for:
- Evaluation modes (CoT vs Direct)
- Dataset descriptions
- Output format
- Batch evaluation
- Comparison with baseline

### Quick Start Guide

See [QUICKSTART_LORA.md](QUICKSTART_LORA.md) for:
- Step-by-step instructions
- Common use cases
- Quick examples
- Memory requirements

## Common Workflows

### Workflow 1: Train and Evaluate Single Model

```bash
# 1. Train
python lora_finetune.py \
  --model_name Qwen/Qwen2-7B \
  --dataset_name folio \
  --output_dir ./lora_qwen2_folio

# 2. Evaluate
python lora_evaluation.py \
  --base_model Qwen/Qwen2-7B \
  --lora_model ./lora_qwen2_folio/final_model \
  --dataset_name FOLIO \
  --split dev \
  --mode CoT \
  --trained_model
```

### Workflow 2: Compare Multiple Models

```bash
# Train multiple models
for MODEL in "Qwen/Qwen2-7B" "google/gemma-7b"; do
  MODEL_NAME=$(echo $MODEL | sed 's/.*\///')
  python lora_finetune.py \
    --model_name $MODEL \
    --dataset_name folio \
    --output_dir ./lora_${MODEL_NAME}_folio
done

# Evaluate all models
for MODEL_DIR in lora_*_folio; do
  BASE_MODEL=$(echo $MODEL_DIR | sed 's/lora_//' | sed 's/_folio//')
  python lora_evaluation.py \
    --base_model $BASE_MODEL \
    --lora_model ./$MODEL_DIR/final_model \
    --dataset_name FOLIO \
    --split dev \
    --mode CoT \
    --trained_model
done
```

### Workflow 3: Cross-Dataset Evaluation

```bash
# Train on FOLIO
python lora_finetune.py \
  --model_name Qwen/Qwen2-7B \
  --dataset_name folio \
  --output_dir ./lora_qwen2_folio

# Evaluate on all datasets (test generalization)
python batch_evaluate_lora.py \
  --base_model Qwen/Qwen2-7B \
  --lora_model ./lora_qwen2_folio/final_model \
  --datasets FOLIO ProofWriter ProntoQA \
  --modes CoT
```

### Workflow 4: Hyperparameter Search

```bash
# Try different LoRA ranks
for RANK in 8 16 32 64; do
  python lora_finetune.py \
    --model_name Qwen/Qwen2-7B \
    --dataset_name folio \
    --output_dir ./lora_qwen2_r${RANK} \
    --lora_r $RANK \
    --lora_alpha $((RANK * 2))
done

# Evaluate each
for DIR in lora_qwen2_r*; do
  python lora_evaluation.py \
    --base_model Qwen/Qwen2-7B \
    --lora_model ./$DIR/final_model \
    --dataset_name FOLIO \
    --split dev \
    --mode CoT \
    --trained_model
done
```

## Performance Expectations

### Training Performance

| Model | Dataset | Time (single GPU) | VRAM (4-bit) |
|-------|---------|-------------------|--------------|
| 7B    | FOLIO   | 2-4 hours        | 6-8 GB       |
| 7B    | ProofWriter-5k | 3-6 hours   | 6-8 GB       |
| 8B    | FOLIO   | 3-5 hours        | 7-9 GB       |

### Evaluation Speed

- ~10-20 examples/minute (depending on model and generation length)
- CoT mode slower than Direct (longer outputs)
- Full FOLIO dev set (~200 examples): 10-20 minutes

## Tips and Best Practices

### Training Tips

1. **Start small**: Use `--num_epochs 1` for initial testing
2. **Monitor validation**: Check loss doesn't increase
3. **Adjust batch size**: Based on your VRAM
4. **Learning rate**: 2e-4 works well for most models
5. **LoRA rank**: 16-32 is usually sufficient

### Evaluation Tips

1. **Use `--trained_model`**: Simplifies prompt format
2. **Temperature 0**: For reproducible results
3. **Test first 10**: Use `--end 10` to verify setup
4. **Check verbose**: Use `--verbose` for debugging
5. **Batch evaluate**: Save time with `batch_evaluate_lora.py`

### Debugging Tips

1. **Check data format**: Inspect training/eval data files
2. **Verify paths**: Ensure dataset paths are correct
3. **Test inference**: Use `lora_inference.py` interactively
4. **Monitor memory**: Use `nvidia-smi` to check VRAM
5. **Reduce batch size**: If OOM errors occur

## Troubleshooting

### Common Issues

**Training OOM**
```bash
# Solution: Reduce batch size, increase gradient accumulation
--batch_size 1 --gradient_accumulation_steps 16
```

**Can't access model**
```bash
# Solution: Login to Hugging Face and get approval
huggingface-cli login
```

**Low accuracy**
```bash
# Solution: Check you're using --trained_model flag
# Try longer training (more epochs)
# Verify data format matches training
```

**Slow evaluation**
```bash
# Solution: Use Direct mode instead of CoT
# Reduce max_new_tokens
# Use batch_evaluate for parallel processing
```

## Resources

- **Training**: See [README_LORA.md](README_LORA.md)
- **Evaluation**: See [README_LORA_EVALUATION.md](README_LORA_EVALUATION.md)
- **Quick Start**: See [QUICKSTART_LORA.md](QUICKSTART_LORA.md)
- **Examples**: See `*_examples.sh` files

## Support

For issues or questions:
1. Check the detailed READMEs
2. Review example scripts
3. Try with `--verbose` flag
4. Check Hugging Face model pages for model-specific requirements

## Citation

If you use this pipeline in your research, please cite the original datasets:

- **FOLIO**: [FOLIO Dataset](https://github.com/yale-lily/FOLIO)
- **ProofWriter**: [ProofWriter Dataset](https://allenai.org/data/proofwriter)

## License

This pipeline is provided for research and educational purposes. Check individual model licenses on Hugging Face.
