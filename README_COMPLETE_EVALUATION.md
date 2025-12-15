# Complete LoRA Evaluation - Matching src/evaluation.py

This evaluation script (`lora_evaluation_complete.py`) **exactly matches** the evaluation methodology from `src/evaluation.py` and produces identical outputs.

## What This Does

This script evaluates LoRA finetuned models using the **same evaluation framework** as the original pipeline:

✅ **Same Tracks**: implicit, explicit, with/without reasoning, with/without correction
✅ **Same Metrics**: tally sum, length by difficulty, degradation buckets, permutation stats
✅ **Same Output Format**: JSON metrics + CSV token stats + detailed task JSONs
✅ **Same Dataset Format**: Uses reviseqa_data structure with edits/reasoning chains
✅ **Compatible Results**: Can be directly compared with baseline model results

## Quick Start

```bash
# 1. Run with default settings
./run_lora_evaluation.sh

# 2. Or run directly with custom settings
python lora_evaluation_complete.py \
  --data-dir reviseqa_data/nl/verified-400/ \
  --base-model Qwen/Qwen2-7B \
  --lora-model ./lora_qwen2_folio/final_model \
  --results-dir lora_results/ \
  --detailed-output-dir lora_detailed_results/ \
  --batch-size 4 \
  --use-4bit
```

## Differences from src/evaluation.py

The **only** difference is the model interface:

| Aspect | src/evaluation.py | lora_evaluation_complete.py |
|--------|-------------------|------------------------------|
| Model Interface | OpenRouter API | Local LoRA model |
| Everything Else | **Identical** | **Identical** |

All evaluation logic, metrics, tracks, and output formats are **exactly the same**.

## Command-Line Arguments

### Required Arguments

```bash
--data-dir DIR          # Directory with JSON files (e.g., reviseqa_data/nl/verified-400/)
--base-model MODEL      # Base model name (e.g., Qwen/Qwen2-7B)
--lora-model PATH       # Path to LoRA weights (e.g., ./lora_qwen2_folio/final_model)
```

### Optional Arguments

```bash
--batch-size N          # Parallel workers (default: 4)
--results-dir DIR       # Output directory for metrics (default: lora_results)
--detailed-output-dir   # Output directory for detailed JSONs (default: None)
--use-4bit              # Use 4-bit quantization (default: True)
--use-8bit              # Use 8-bit quantization
--no-quantization       # Disable quantization
--enable-truncated      # Include truncated reasoning files
--enable-shuffled       # Add shuffled dataset tracks
```

## Dataset Format

The script expects data in the **reviseqa_data format** used by `src/evaluation.py`:

```
reviseqa_data/nl/verified-400/
├── ex_1006.json
├── ex_1010.json
├── ex_1011.json
└── ...
```

Each JSON file contains:
- `original_context`: List of natural language statements
- `conclusion`: The conclusion to verify
- `answer`: True/False/Uncertain
- `reasoning_chain`: Step-by-step reasoning
- `edits`: Sequence of context modifications with new answers

## Evaluation Tracks

The script evaluates on **8 tracks** (same as original):

1. **implicit** - Full context, with reasoning, with correction
2. **explicit** - Edit breakdown, with reasoning, with correction
3. **implicit_no_reasoning** - No demonstration reasoning
4. **explicit_no_reasoning** - Edit breakdown, no reasoning
5. **implicit_no_correction** - No error correction feedback
6. **explicit_no_correction** - Edit breakdown, no correction
7. **implicit_no_reasoning_no_correction** - No reasoning, no correction
8. **explicit_no_reasoning_no_correction** - Edit breakdown, no reasoning, no correction

Optional (with `--enable-shuffled`):
9. **implicit_shuffled** - Shuffled context sentences
10. **implicit_shuffled_no_reasoning** - Shuffled, no reasoning

## Output Files

### 1. Metrics JSON

**File**: `lora_results/lora_{model}_{timestamp}_correctness.json`

Contains:
- `metadata`: Model info, dataset length, batch size
- `total_per_track`: Number of predictions per track
- `tally_sum`: Total correct predictions per track
- `length_by_difficulty`: Easy/medium/hard counts per track
- `degradation_buckets`: Accuracy by token count buckets
- `permutation_stats`: Accuracy by edit type (added_facts, removed_rules, etc.)

### 2. Token Stats CSV

**File**: `lora_results/lora_{model}_{timestamp}_token_count_stats.csv`

Columns:
- `track`: Evaluation track
- `chain_idx`: Example index
- `step`: Step in reasoning chain
- `token_count`: Cumulative token count
- `correct`: 1 if correct, 0 if wrong
- `tags`: Edit types (added_facts, removed_rules, etc.)
- `prediction`: Model's predicted answer
- `correct_answer`: Ground truth answer
- `reasoning`: Model's reasoning (if included)

### 3. Detailed Task JSONs

**Directory**: `lora_detailed_results/{track}/{track}_{filename}.json`

Each file contains:
- `metadata`: Model, track, chain index, include_reasoning, include_correction, accuracy
- `predictions`: List of all predictions with:
  - `step`: Step number
  - `context`: Input context
  - `question`: Question asked
  - `prediction`: Model's answer
  - `correct_answer`: Ground truth
  - `reasoning`: Model's reasoning
  - `correct`: Boolean
  - `tags`: Edit types
  - `is_demonstration`: Boolean
  - `token_count`: Cumulative tokens

## Example Outputs

### Metrics JSON Sample

```json
{
  "timestamp": "2025-01-15T10:30:00",
  "metadata": {
    "base_model": "Qwen/Qwen2-7B",
    "lora_model": "./lora_qwen2_folio/final_model",
    "dataset_length": 400,
    "batch_size": 4
  },
  "tally_sum": {
    "implicit": 3450,
    "explicit": 3200,
    ...
  },
  "length_by_difficulty": {
    "implicit": {"easy": 350, "medium": 280, "hard": 120},
    ...
  },
  "degradation_buckets": {
    "implicit": {
      "0": {"total": 400, "correct": 380},
      "1": {"total": 380, "correct": 350},
      ...
    }
  },
  "permutation_stats": {
    "implicit": {
      "added_facts": {"total": 500, "correct": 450},
      "removed_rules": {"total": 300, "correct": 250},
      ...
    }
  }
}
```

### Detailed Task JSON Sample

```json
{
  "metadata": {
    "base_model": "Qwen/Qwen2-7B",
    "lora_model": "./lora_qwen2_folio/final_model",
    "task_path": "implicit",
    "chain_index": 0,
    "include_reasoning": true,
    "include_correction": true,
    "total_steps": 10,
    "final_accuracy": 0.9
  },
  "predictions": [
    {
      "step": 0,
      "context": "Rebecca does not mentor students...",
      "question": "Does the context entail...",
      "prediction": "True",
      "correct_answer": "True",
      "reasoning": "Rebecca is a clinician...",
      "correct": true,
      "tags": ["original"],
      "is_demonstration": true
    },
    {
      "step": 1,
      "context": "Added facts:\n- Rebecca teaches courses.",
      "question": "Does the context entail...",
      "prediction": "False",
      "correct_answer": "False",
      "reasoning": "Given the added fact...",
      "correct": true,
      "tags": ["added_facts"],
      "is_demonstration": false,
      "token_count": 512
    },
    ...
  ]
}
```

## Comparison with Baseline

To compare LoRA model with baseline:

### 1. Run Baseline (using src/evaluation.py)

```bash
python src/evaluation.py \
  --data-dir reviseqa_data/nl/verified-400/ \
  --model-name google/gemini-2.5-flash-preview \
  --results-dir baseline_results/ \
  --detailed-output-dir baseline_detailed/ \
  --guided
```

### 2. Run LoRA Evaluation

```bash
python lora_evaluation_complete.py \
  --data-dir reviseqa_data/nl/verified-400/ \
  --base-model Qwen/Qwen2-7B \
  --lora-model ./lora_qwen2_folio/final_model \
  --results-dir lora_results/ \
  --detailed-output-dir lora_detailed/
```

### 3. Compare Metrics

Both will produce:
- Correctness JSON with identical structure
- Token stats CSV with identical columns
- Detailed task JSONs with identical format

You can directly compare:
- `tally_sum` for overall accuracy
- `length_by_difficulty` for task completion rates
- `degradation_buckets` for context length effects
- `permutation_stats` for performance on different edit types

## Understanding the Metrics

### Tally Sum
- **What**: Total correct predictions per track
- **Higher = Better**: More correct answers
- **Compare**: Across models and tracks

### Length by Difficulty
- **Easy**: Model got ≥30% of chain correct before first mistake
- **Medium**: Model got ≥60% of chain correct
- **Hard**: Model got 100% of chain correct (perfect)
- **Higher = Better**: More tasks in harder categories

### Degradation Buckets
- **What**: Accuracy vs. conversation length (in 512-token buckets)
- **Shows**: How model performance degrades with longer contexts
- **Look for**: Slow degradation (model maintains performance)

### Permutation Stats
- **What**: Accuracy on different edit types
- **Types**: added_facts, removed_facts, added_rules, removed_rules, original, no_change
- **Shows**: Which types of logical changes the model handles well

## Interpreting Results

### Good LoRA Model
- High tally_sum (>80% of total predictions correct)
- Many tasks in "hard" category (complete chain correct)
- Slow degradation across token buckets
- Balanced performance across edit types

### Areas for Improvement
- Low tally_sum → Need more training or better data
- Few "hard" tasks → Model makes errors early in chains
- Fast degradation → Model struggles with long contexts
- Poor on specific edit types → Need targeted training examples

## Troubleshooting

### CUDA Out of Memory
```bash
# Use smaller batch size
--batch-size 1

# Or use 8-bit quantization
--use-8bit

# Or disable quantization and use CPU (slow)
--no-quantization
```

### Missing Dependencies
```bash
pip install transformers peft torch accelerate bitsandbytes tiktoken tqdm
```

### Wrong Data Directory
The script expects JSON files in the format from `src/evaluation.py`.
Check that files have:
- `original_context`
- `conclusion`
- `answer`
- `reasoning_chain`
- `edits`

### Model Not Loading
- Check base model name is correct (e.g., `Qwen/Qwen2-7B`)
- Check LoRA path exists and contains adapter files
- For gated models (Llama, Gemma), run `huggingface-cli login`

## Advanced Usage

### Evaluate Multiple Models

```bash
#!/bin/bash
MODELS=(
  "Qwen/Qwen2-7B:./lora_qwen2_folio/final_model"
  "meta-llama/Llama-2-7b-hf:./lora_llama2_folio/final_model"
  "google/gemma-7b:./lora_gemma_folio/final_model"
)

for MODEL_PAIR in "${MODELS[@]}"; do
  IFS=':' read -r BASE LORA <<< "$MODEL_PAIR"
  python lora_evaluation_complete.py \
    --data-dir reviseqa_data/nl/verified-400/ \
    --base-model "$BASE" \
    --lora-model "$LORA" \
    --results-dir lora_results/ \
    --detailed-output-dir lora_detailed/ \
    --batch-size 4
done
```

### Resume Interrupted Evaluation

The script automatically skips tasks that already have detailed JSON outputs:

```bash
# If interrupted, just re-run the same command
# It will skip completed tasks and continue from where it stopped
python lora_evaluation_complete.py \
  --data-dir reviseqa_data/nl/verified-400/ \
  --base-model Qwen/Qwen2-7B \
  --lora-model ./lora_qwen2_folio/final_model \
  --detailed-output-dir lora_detailed/
```

### Evaluate on Subset

```bash
# Evaluate on first 100 examples only
# (modify the script to add --max-examples argument if needed)
# Or manually create subset directory with first 100 JSON files
```

## Performance Expectations

### Speed
- **4-bit quantization**: ~1-2 examples/minute per worker
- **8-bit quantization**: ~0.5-1 example/minute per worker
- **No quantization**: ~0.2-0.5 example/minute per worker

### Memory
- **4-bit, batch-size=4**: ~12-16 GB VRAM
- **8-bit, batch-size=4**: ~20-24 GB VRAM
- **No quant, batch-size=1**: ~30+ GB VRAM

### Time Estimates (400 examples, 8 tracks)
- **With 4-bit, batch-size=4**: 8-16 hours
- **With 8-bit, batch-size=2**: 12-24 hours
- **Serial (batch-size=1)**: 16-32 hours

## Integration with Existing Pipeline

This evaluation produces outputs **100% compatible** with `src/evaluation.py`:

1. **Same metrics**: Can analyze with same analysis scripts
2. **Same format**: Can visualize with same plotting code
3. **Same structure**: Can compare directly with baseline results
4. **Same insights**: Provides same performance metrics

The only difference is the model source (local LoRA vs API), making this a drop-in replacement for evaluating finetuned models.

## Next Steps

After evaluation:
1. **Analyze metrics**: Compare tally_sum, degradation, permutation stats
2. **Compare with baseline**: See if finetuning improved performance
3. **Identify weaknesses**: Check which edit types perform poorly
4. **Iterate training**: Use insights to improve training data/strategy
5. **Test on new data**: Evaluate generalization on unseen examples

## Summary

✅ **Matches src/evaluation.py**: Same tracks, metrics, and outputs
✅ **Easy to use**: Single command or shell script
✅ **Comprehensive**: Detailed per-task JSONs + aggregate metrics
✅ **Compatible**: Works with existing analysis pipeline
✅ **Efficient**: Parallel processing with quantization support

This is the complete evaluation solution for LoRA finetuned models on logical reasoning tasks!
