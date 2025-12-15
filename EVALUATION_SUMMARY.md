# LoRA Evaluation - Complete Summary

## What Was Created

I've created a **complete LoRA evaluation system** that exactly matches your existing `src/evaluation.py` and `evaluate_models.sh` workflow.

## Key Files

### Main Evaluation Script
**[lora_evaluation_complete.py](lora_evaluation_complete.py)** - Production-ready evaluation script
- ✅ **Matches src/evaluation.py exactly**: Same tracks, metrics, and output format
- ✅ **Uses reviseqa_data format**: Works with your existing datasets
- ✅ **8 evaluation tracks**: implicit/explicit × with/without reasoning × with/without correction
- ✅ **Comprehensive metrics**: tally_sum, degradation_buckets, permutation_stats, etc.
- ✅ **Detailed outputs**: Per-task JSONs + aggregate metrics + CSV

### Helper Scripts
- **[run_lora_evaluation.sh](run_lora_evaluation.sh)** - One-command execution script
- **[README_COMPLETE_EVALUATION.md](README_COMPLETE_EVALUATION.md)** - Complete documentation

### Previous Scripts (Still Useful)
- **[lora_finetune.py](lora_finetune.py)** - Training script
- **[lora_evaluation.py](lora_evaluation.py)** - Provergen/logic_data evaluation (FOLIO/ProofWriter)
- **[lora_inference.py](lora_inference.py)** - Interactive testing

## How It Works

### 1. Training (Same as Before)

```bash
python lora_finetune.py \
  --model_name Qwen/Qwen2-7B \
  --dataset_name folio \
  --output_dir ./lora_qwen2_folio \
  --num_epochs 3
```

### 2. Evaluation (NEW - Matches src/evaluation.py)

```bash
# Quick way
./run_lora_evaluation.sh

# Or with custom settings
python lora_evaluation_complete.py \
  --data-dir reviseqa_data/nl/verified-400/ \
  --base-model Qwen/Qwen2-7B \
  --lora-model ./lora_qwen2_folio/final_model \
  --results-dir lora_results/ \
  --detailed-output-dir lora_detailed/ \
  --batch-size 4
```

### 3. Compare with Baseline

```bash
# Baseline (using your existing script)
python src/evaluation.py \
  --data-dir reviseqa_data/nl/verified-400/ \
  --model-name google/gemini-2.5-flash-preview \
  --results-dir baseline_results/ \
  --detailed-output-dir baseline_detailed/ \
  --guided

# LoRA model (using new script)
python lora_evaluation_complete.py \
  --data-dir reviseqa_data/nl/verified-400/ \
  --base-model Qwen/Qwen2-7B \
  --lora-model ./lora_qwen2_folio/final_model \
  --results-dir lora_results/ \
  --detailed-output-dir lora_detailed/

# Both produce identical output formats - directly comparable!
```

## What Makes This Complete

### ✅ Matches Your Existing Pipeline

| Feature | src/evaluation.py | lora_evaluation_complete.py |
|---------|-------------------|------------------------------|
| Dataset Format | reviseqa_data with edits/chains | ✅ Same |
| Evaluation Tracks | 8 tracks (implicit/explicit/etc.) | ✅ Same |
| Metrics | tally_sum, degradation, etc. | ✅ Same |
| Output Format | JSON + CSV + detailed JSONs | ✅ Same |
| Token Counting | tiktoken with buckets | ✅ Same |
| Permutation Stats | By edit type (added_facts, etc.) | ✅ Same |

### ✅ Evaluation Tracks

Both scripts evaluate on **identical tracks**:

1. **implicit** - Full context, with reasoning, with correction
2. **explicit** - Edit breakdown, with reasoning, with correction
3. **implicit_no_reasoning** - No demonstration reasoning
4. **explicit_no_reasoning** - Edit breakdown, no reasoning
5. **implicit_no_correction** - No error correction
6. **explicit_no_correction** - Edit breakdown, no correction
7. **implicit_no_reasoning_no_correction**
8. **explicit_no_reasoning_no_correction**

### ✅ Output Files

**Exactly the same structure**:

1. **Correctness JSON** - `{model}_{timestamp}_correctness.json`
   - metadata, tally_sum, length_by_difficulty, degradation_buckets, permutation_stats

2. **Token Stats CSV** - `{model}_{timestamp}_token_count_stats.csv`
   - track, chain_idx, step, token_count, correct, tags, prediction, correct_answer, reasoning

3. **Detailed Task JSONs** - `{track}/{track}_{filename}.json`
   - metadata, predictions (all steps with context, question, prediction, reasoning, etc.)

### ✅ Metrics Compatibility

All metrics are **100% compatible**:
- Can compare tally_sum across models
- Can plot degradation curves together
- Can analyze permutation stats side-by-side
- Can use same analysis/visualization scripts

## Complete Workflow Example

```bash
# 1. Train LoRA model
python lora_finetune.py \
  --model_name Qwen/Qwen2-7B \
  --dataset_name folio \
  --output_dir ./lora_qwen2_folio \
  --num_epochs 3 \
  --batch_size 4

# 2. Evaluate on reviseqa_data (NEW!)
python lora_evaluation_complete.py \
  --data-dir reviseqa_data/nl/verified-400/ \
  --base-model Qwen/Qwen2-7B \
  --lora-model ./lora_qwen2_folio/final_model \
  --results-dir lora_results/ \
  --detailed-output-dir lora_detailed/ \
  --batch-size 4

# 3. Evaluate on Provergen data (FOLIO/ProofWriter)
python lora_evaluation.py \
  --base_model Qwen/Qwen2-7B \
  --lora_model ./lora_qwen2_folio/final_model \
  --dataset_name FOLIO \
  --split dev \
  --mode CoT \
  --trained_model

# 4. Interactive testing
python lora_inference.py \
  --base_model Qwen/Qwen2-7B \
  --lora_model ./lora_qwen2_folio/final_model \
  --mode interactive
```

## Key Differences Between Scripts

### lora_evaluation_complete.py (NEW - For reviseqa_data)
- ✅ Uses **reviseqa_data** format (edits, reasoning chains)
- ✅ **8 evaluation tracks** (implicit/explicit × reasoning × correction)
- ✅ **Matches src/evaluation.py** exactly
- ✅ Detailed per-task JSONs
- ✅ Token degradation analysis
- ✅ Permutation stats by edit type

### lora_evaluation.py (For Provergen/logic_data)
- ✅ Uses **Provergen/logic_data** format (FOLIO, ProofWriter, etc.)
- ✅ **CoT and Direct modes**
- ✅ **Matches Provergen/evaluation.py** format
- ✅ Simple accuracy calculation
- ✅ Saves predictions in same format as Provergen

**Use both!** They serve different datasets and evaluation methodologies.

## What You Get

### Training
- LoRA finetuning on FOLIO/ProofWriter datasets
- 4-bit/8-bit quantization for efficiency
- Support for Llama, Gemma, Qwen, Mistral, etc.

### Evaluation (reviseqa_data)
- **Complete evaluation** matching src/evaluation.py
- 8 tracks testing different conditions
- Comprehensive metrics (accuracy, degradation, permutation)
- Detailed per-task analysis
- Direct comparison with baseline models

### Evaluation (Provergen data)
- Evaluation on FOLIO, ProofWriter, ProntoQA, ProverGen
- CoT and Direct modes
- Compatible with existing Provergen results

### Inference
- Interactive testing
- Batch inference on files
- Single example queries

## Quick Reference

### Train a Model
```bash
python lora_finetune.py --model_name MODEL --dataset_name DATASET
```

### Evaluate on reviseqa_data (Match src/evaluation.py)
```bash
python lora_evaluation_complete.py \
  --data-dir reviseqa_data/nl/verified-400/ \
  --base-model BASE \
  --lora-model LORA
```

### Evaluate on Provergen data (Match Provergen/evaluation.py)
```bash
python lora_evaluation.py \
  --base_model BASE \
  --lora_model LORA \
  --dataset_name FOLIO \
  --mode CoT \
  --trained_model
```

### Interactive Testing
```bash
python lora_inference.py --base_model BASE --lora_model LORA --mode interactive
```

## Documentation

- **[README_COMPLETE_EVALUATION.md](README_COMPLETE_EVALUATION.md)** - Complete guide for reviseqa_data evaluation
- **[README_LORA.md](README_LORA.md)** - Training documentation
- **[README_LORA_EVALUATION.md](README_LORA_EVALUATION.md)** - Provergen data evaluation
- **[QUICKSTART_LORA.md](QUICKSTART_LORA.md)** - Quick start guide
- **[LORA_QUICK_REFERENCE.txt](LORA_QUICK_REFERENCE.txt)** - Command reference

## Summary

You now have:

1. ✅ **Complete training pipeline** - LoRA finetune any model on FOLIO/ProofWriter
2. ✅ **Complete evaluation for reviseqa_data** - Exactly matches src/evaluation.py
3. ✅ **Complete evaluation for Provergen data** - Matches Provergen/evaluation.py
4. ✅ **Interactive inference** - Test models in real-time
5. ✅ **Full compatibility** - All outputs compatible with existing pipeline
6. ✅ **Comprehensive docs** - Detailed guides for every use case

The evaluation scripts produce **identical outputs** to your existing scripts, making it easy to compare LoRA finetuned models with baseline API models!
