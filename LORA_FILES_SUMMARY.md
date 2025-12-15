# LoRA Finetuning and Evaluation - Complete File Summary

## Created Files Overview

This document lists all the files created for LoRA finetuning and evaluation.

## Core Scripts (6 files)

### 1. Training Scripts

| File | Purpose | Usage |
|------|---------|-------|
| `lora_finetune.py` | Main training script | Train models on FOLIO/ProofWriter |
| `merge_lora_weights.py` | Merge LoRA into base model | Deploy merged models |

### 2. Evaluation Scripts

| File | Purpose | Usage |
|------|---------|-------|
| `lora_evaluation.py` | Evaluate on test sets (compatible with evaluation.py) | Run same evaluations as original pipeline |
| `evaluate_lora_model.py` | Standalone evaluation with metrics | Quick accuracy checks |
| `batch_evaluate_lora.py` | Batch evaluation runner | Evaluate on multiple datasets/modes |

### 3. Inference Scripts

| File | Purpose | Usage |
|------|---------|-------|
| `lora_inference.py` | Interactive and batch inference | Test models interactively |

## Example Scripts (2 files)

| File | Purpose |
|------|---------|
| `finetune_examples.sh` | 12 training examples for different models/configs |
| `lora_evaluation_examples.sh` | 12 evaluation examples for different datasets/modes |

## Documentation (5 files)

| File | Purpose | When to Read |
|------|---------|--------------|
| `README_LORA_COMPLETE.md` | **Complete guide** - Start here | Overview and workflows |
| `QUICKSTART_LORA.md` | Quick start guide | Getting started quickly |
| `README_LORA.md` | Training documentation | Detailed training info |
| `README_LORA_EVALUATION.md` | Evaluation documentation | Detailed evaluation info |
| `LORA_FILES_SUMMARY.md` | This file | File reference |

## Configuration Files (1 file)

| File | Purpose |
|------|---------|
| `requirements_lora.txt` | Python dependencies for LoRA training/evaluation |

## Total: 14 New Files

```
Training:
  - lora_finetune.py
  - merge_lora_weights.py
  - finetune_examples.sh

Evaluation:
  - lora_evaluation.py
  - evaluate_lora_model.py
  - batch_evaluate_lora.py
  - lora_evaluation_examples.sh

Inference:
  - lora_inference.py

Documentation:
  - README_LORA_COMPLETE.md
  - QUICKSTART_LORA.md
  - README_LORA.md
  - README_LORA_EVALUATION.md
  - LORA_FILES_SUMMARY.md

Config:
  - requirements_lora.txt
```

## Quick Reference

### To Train a Model
```bash
python lora_finetune.py --model_name MODEL --dataset_name DATASET
```
See: `README_LORA.md` or `finetune_examples.sh`

### To Evaluate a Model
```bash
python lora_evaluation.py --base_model BASE --lora_model LORA --dataset_name DATASET
```
See: `README_LORA_EVALUATION.md` or `lora_evaluation_examples.sh`

### To Test Interactively
```bash
python lora_inference.py --base_model BASE --lora_model LORA --mode interactive
```
See: `README_LORA.md`

### To Merge Weights
```bash
python merge_lora_weights.py --base_model BASE --lora_model LORA --output_path OUTPUT
```
See: `README_LORA.md`

## File Relationships

```
Training Flow:
  lora_finetune.py → [LoRA weights] → evaluate_lora_model.py
                                    → lora_evaluation.py
                                    → lora_inference.py
                                    → merge_lora_weights.py

Evaluation Flow:
  lora_evaluation.py → [results.json] → (compare with baseline)
  batch_evaluate_lora.py → [multiple results] → (analysis)

Documentation Flow:
  README_LORA_COMPLETE.md (overview)
    ├── QUICKSTART_LORA.md (quick start)
    ├── README_LORA.md (training details)
    └── README_LORA_EVALUATION.md (evaluation details)
```

## Where to Start

### New Users
1. Read: `README_LORA_COMPLETE.md` (overview)
2. Read: `QUICKSTART_LORA.md` (step-by-step)
3. Run: Examples from `finetune_examples.sh`

### Training Focus
1. Read: `README_LORA.md`
2. Review: `finetune_examples.sh`
3. Run: `lora_finetune.py`

### Evaluation Focus
1. Read: `README_LORA_EVALUATION.md`
2. Review: `lora_evaluation_examples.sh`
3. Run: `lora_evaluation.py` or `batch_evaluate_lora.py`

### Interactive Testing
1. Read: Section in `README_LORA.md`
2. Run: `lora_inference.py --mode interactive`

## All Supported Models

The scripts support any causal language model, with auto-detection for:
- Llama (2, 3, 3.1, etc.)
- Gemma (1, 2)
- Qwen (2, 2.5)
- Mistral
- And more...

## All Supported Datasets

### Training Datasets
- FOLIO (in `Provergen/training_data/folio.json`)
- ProofWriter-5k (in `Provergen/training_data/proofwriter-5000.json`)
- ProofWriter-Full (in `Provergen/training_data/proofwriter-full.json`)

### Evaluation Datasets
- FOLIO (in `Provergen/logic_data/FOLIO/`)
- ProofWriter (in `Provergen/logic_data/ProofWriter/`)
- ProntoQA (in `Provergen/logic_data/ProntoQA/`)
- ProverGen (in `Provergen/logic_data/ProverQA/`)

## Key Features Summary

✅ Multi-model support (Llama, Gemma, Qwen, Mistral)
✅ Efficient 4-bit/8-bit quantization
✅ Both FOLIO and ProofWriter datasets
✅ CoT and Direct evaluation modes
✅ Compatible with existing evaluation.py pipeline
✅ Interactive testing mode
✅ Batch evaluation across datasets
✅ LoRA weight merging for deployment
✅ Comprehensive documentation
✅ Ready-to-run examples

## Installation

```bash
pip install -r requirements_lora.txt
huggingface-cli login  # For gated models
```

## Complete Example Workflow

```bash
# 1. Train
python lora_finetune.py \
  --model_name Qwen/Qwen2-7B \
  --dataset_name folio \
  --output_dir ./lora_qwen2_folio

# 2. Evaluate (same as evaluation.py)
python lora_evaluation.py \
  --base_model Qwen/Qwen2-7B \
  --lora_model ./lora_qwen2_folio/final_model \
  --dataset_name FOLIO \
  --split dev \
  --mode CoT \
  --trained_model

# 3. Test interactively
python lora_inference.py \
  --base_model Qwen/Qwen2-7B \
  --lora_model ./lora_qwen2_folio/final_model \
  --mode interactive

# 4. Merge for deployment
python merge_lora_weights.py \
  --base_model Qwen/Qwen2-7B \
  --lora_model ./lora_qwen2_folio/final_model \
  --output_path ./merged_qwen2_folio
```

## Next Steps

After reviewing this summary:
1. Read `README_LORA_COMPLETE.md` for complete overview
2. Follow `QUICKSTART_LORA.md` for step-by-step guide
3. Review example scripts for your use case
4. Start training and evaluating!
