# LoRA Model Evaluation Guide

This guide explains how to evaluate LoRA finetuned models using the same evaluation framework as the original `evaluation.py` script.

## Overview

The evaluation scripts allow you to:
- Evaluate LoRA finetuned models on FOLIO, ProofWriter, ProntoQA, and ProverGen datasets
- Use both CoT (Chain-of-Thought) and Direct evaluation modes
- Run evaluations in the same format as the original evaluation pipeline
- Batch evaluate across multiple datasets and modes
- Compare finetuned models against baseline results

## Files

- **[lora_evaluation.py](lora_evaluation.py)** - Main evaluation script (compatible with evaluation.py format)
- **[batch_evaluate_lora.py](batch_evaluate_lora.py)** - Batch evaluation across datasets
- **[lora_evaluation_examples.sh](lora_evaluation_examples.sh)** - Example commands

## Quick Start

### 1. Single Dataset Evaluation

Evaluate a LoRA model on FOLIO dev set with CoT:

```bash
python lora_evaluation.py \
  --base_model meta-llama/Llama-2-7b-hf \
  --lora_model ./lora_llama2_folio/final_model \
  --dataset_name FOLIO \
  --split dev \
  --mode CoT \
  --trained_model \
  --output_dir lora_results/
```

### 2. Batch Evaluation

Evaluate on all datasets and modes:

```bash
python batch_evaluate_lora.py \
  --base_model meta-llama/Llama-2-7b-hf \
  --lora_model ./lora_llama2_folio/final_model \
  --datasets FOLIO ProofWriter ProntoQA \
  --splits dev test \
  --modes CoT Direct \
  --output_dir lora_results/
```

## Evaluation Modes

### CoT (Chain-of-Thought)
- Model provides reasoning steps before the answer
- System prompt: "Your answer should be in JSON format with keys: reasoning, answer"
- Max tokens: 1024 (default)
- Use for: Better understanding of model reasoning

### Direct
- Model provides only the final answer
- System prompt: "Your answer should be in JSON format with key: answer"
- Max tokens: 128 (default)
- Use for: Faster evaluation, testing answer accuracy

## Available Datasets

### FOLIO
- Logical reasoning with natural language contexts
- **Splits**: train, dev
- **Path**: `Provergen/logic_data/FOLIO/`

### ProofWriter
- Formal logic reasoning with rules and facts
- **Splits**: train, dev, test
- **Path**: `Provergen/logic_data/ProofWriter/`

### ProntoQA
- Question answering with logical reasoning
- **Splits**: dev
- **Path**: `Provergen/logic_data/ProntoQA/`

### ProverGen
- Custom logical reasoning dataset
- **Splits**: easy, medium, hard
- **Path**: `Provergen/logic_data/ProverQA/`

## Command-Line Arguments

### Model Arguments

```bash
--base_model        # Base model name (required)
--lora_model        # Path to LoRA weights (required)
--use_4bit          # Use 4-bit quantization (default: True)
--use_8bit          # Use 8-bit quantization
--no_quantization   # Disable quantization
--merge_weights     # Merge LoRA into base model before eval
```

### Dataset Arguments

```bash
--dataset_name      # Dataset to evaluate (FOLIO, ProofWriter, etc.)
--split             # Dataset split (dev, test, train)
--mode              # Evaluation mode (CoT or Direct)
--data_path         # Path to dataset directory
--demonstration_path # Path to few-shot examples
```

### Evaluation Arguments

```bash
--output_dir        # Output directory for results
--start             # Start index (default: 0)
--end               # End index (default: all)
--trained_model     # Use simple prompt for finetuned models
--temperature       # Sampling temperature (default: 0.0)
--max_new_tokens    # Max tokens to generate
--verbose           # Print detailed output
```

## Output Format

Results are saved in JSON format compatible with the original evaluation.py:

```json
[
  {
    "id": "FOLIO_dev_0",
    "context": [...],  // Full prompt as messages
    "question": "...",
    "label": "C",  // Ground truth answer
    "model_answer": "{\"reasoning\": \"...\", \"answer\": \"C\"}"
  },
  ...
]
```

Output filename format:
```
{mode}_{dataset}_{split}_{model_name}_{start}-{end}.json
```

Example: `CoT_FOLIO_dev_final_model_0-100.json`

## Examples

### Evaluate on FOLIO (CoT and Direct)

```bash
# CoT mode
python lora_evaluation.py \
  --base_model meta-llama/Llama-2-7b-hf \
  --lora_model ./lora_llama2_folio/final_model \
  --dataset_name FOLIO \
  --split dev \
  --mode CoT \
  --trained_model

# Direct mode
python lora_evaluation.py \
  --base_model meta-llama/Llama-2-7b-hf \
  --lora_model ./lora_llama2_folio/final_model \
  --dataset_name FOLIO \
  --split dev \
  --mode Direct \
  --trained_model
```

### Evaluate on ProofWriter

```bash
python lora_evaluation.py \
  --base_model Qwen/Qwen2-7B \
  --lora_model ./lora_qwen2_proofwriter/final_model \
  --dataset_name ProofWriter \
  --split test \
  --mode CoT \
  --trained_model
```

### Evaluate First 100 Examples

```bash
python lora_evaluation.py \
  --base_model meta-llama/Llama-2-7b-hf \
  --lora_model ./lora_llama2_folio/final_model \
  --dataset_name FOLIO \
  --split dev \
  --mode CoT \
  --trained_model \
  --start 0 \
  --end 100
```

### Batch Evaluate All Datasets

```bash
python batch_evaluate_lora.py \
  --base_model meta-llama/Llama-2-7b-hf \
  --lora_model ./lora_llama2_folio/final_model \
  --datasets FOLIO ProofWriter ProntoQA \
  --modes CoT Direct
```

### Evaluate Multiple Models

```bash
#!/bin/bash
MODELS=(
  "meta-llama/Llama-2-7b-hf:./lora_llama2_folio/final_model"
  "Qwen/Qwen2-7B:./lora_qwen2_folio/final_model"
  "google/gemma-7b:./lora_gemma_folio/final_model"
)

for MODEL_PAIR in "${MODELS[@]}"; do
  IFS=':' read -r BASE LORA <<< "$MODEL_PAIR"
  python lora_evaluation.py \
    --base_model "$BASE" \
    --lora_model "$LORA" \
    --dataset_name FOLIO \
    --split dev \
    --mode CoT \
    --trained_model \
    --output_dir lora_results/
done
```

## Comparing with Baseline

To compare LoRA model performance with baseline:

1. **Run baseline evaluation** (using original evaluation.py):
```bash
python Provergen/evaluation.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --dataset_name FOLIO \
  --split dev \
  --mode CoT \
  --output_dir baseline_results/
```

2. **Run LoRA evaluation**:
```bash
python lora_evaluation.py \
  --base_model meta-llama/Llama-2-7b-hf \
  --lora_model ./lora_llama2_folio/final_model \
  --dataset_name FOLIO \
  --split dev \
  --mode CoT \
  --trained_model \
  --output_dir lora_results/
```

3. **Compare results**: Both will output accuracy and save predictions in the same format

## Accuracy Calculation

The script automatically computes accuracy by:
1. Extracting the predicted answer from model output (parsing JSON or text)
2. Comparing with ground truth label
3. Calculating percentage correct

Accuracy is printed at the end of evaluation:
```
================================================================================
Evaluation Results
================================================================================
Dataset: FOLIO
Split: dev
Mode: CoT
Accuracy: 0.8523 (189/222)
================================================================================
```

## Tips for Best Results

1. **Use `--trained_model` flag**: Simplifies prompt format for finetuned models
2. **Set `--temperature 0.0`**: For deterministic, reproducible results
3. **Start with small batches**: Use `--end 10` to test setup before full evaluation
4. **Use `--verbose`**: To debug prompt formatting and responses
5. **Match training format**: Use same prompt style as during training

## Troubleshooting

### Low Accuracy

- Check that `--trained_model` flag is used (simplifies prompt)
- Verify the LoRA model was trained on similar data
- Try both CoT and Direct modes to see which performs better
- Examine verbose output to see if model is following format

### Out of Memory

- Use `--use_4bit` (default)
- Reduce batch processing or use `--start` and `--end` to split evaluation
- Use `--merge_weights` cautiously (may use more memory)

### Wrong Output Format

- Model may need better prompting - check verbose output
- Consider retraining with more emphasis on output format
- Some models may need their native chat template

### Can't Find Dataset

- Check `--data_path` points to correct directory
- Default is `Provergen/logic_data/`
- Verify dataset files exist: `ls Provergen/logic_data/FOLIO/`

## Integration with Existing Pipeline

The LoRA evaluation outputs are compatible with the original evaluation pipeline:

1. Results are in the same JSON format
2. Filenames follow the same convention
3. Can be processed by existing analysis scripts
4. Accuracy metrics are computed identically

This allows you to:
- Compare LoRA models directly with baseline results
- Use existing visualization/analysis tools
- Integrate into existing evaluation workflows

## Next Steps

After evaluation:
1. Analyze results in `lora_results/` directory
2. Compare accuracy across different models and datasets
3. Use results to identify areas for improvement
4. Fine-tune hyperparameters based on evaluation performance
5. Consider ensemble methods combining multiple models

For more details on training, see [README_LORA.md](README_LORA.md)
