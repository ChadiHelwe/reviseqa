# LoRA Finetuning for Logical Reasoning Models

This directory contains scripts for LoRA (Low-Rank Adaptation) finetuning of large language models on the FOLIO and ProofWriter datasets for logical reasoning tasks.

## Features

- **Multi-model support**: Llama, Gemma, Qwen, Mistral, and other causal language models
- **Efficient training**: 4-bit/8-bit quantization using bitsandbytes
- **LoRA**: Parameter-efficient finetuning with customizable rank and alpha
- **Two datasets**: FOLIO and ProofWriter for logical reasoning
- **Easy to use**: Simple command-line interface with sensible defaults

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements_lora.txt
```

2. Make sure you have access to the models you want to finetune (e.g., Hugging Face token for gated models like Llama).

## Quick Start

### Finetune Llama-2-7b on FOLIO

```bash
python lora_finetune.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --dataset_name folio \
  --output_dir ./lora_llama2_folio \
  --num_epochs 3
```

### Finetune Gemma-7b on ProofWriter

```bash
python lora_finetune.py \
  --model_name google/gemma-7b \
  --dataset_name proofwriter \
  --output_dir ./lora_gemma_proofwriter \
  --num_epochs 3
```

### Finetune Qwen2-7B on FOLIO

```bash
python lora_finetune.py \
  --model_name Qwen/Qwen2-7B \
  --dataset_name folio \
  --output_dir ./lora_qwen2_folio \
  --num_epochs 3
```

## Command-Line Arguments

### Model Arguments

- `--model_name`: Model name or path (required)
  - Examples: `meta-llama/Llama-2-7b-hf`, `google/gemma-7b`, `Qwen/Qwen2-7B`
- `--use_4bit`: Use 4-bit quantization (default: True)
- `--use_8bit`: Use 8-bit quantization
- `--no_quantization`: Disable quantization (requires more VRAM)

### LoRA Arguments

- `--lora_r`: LoRA rank (default: 16)
- `--lora_alpha`: LoRA alpha (default: 32)
- `--lora_dropout`: LoRA dropout (default: 0.05)
- `--target_modules`: Target modules for LoRA (auto-detected if not provided)

### Training Arguments

- `--num_epochs`: Number of training epochs (default: 3)
- `--batch_size`: Training batch size per device (default: 4)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 4)
- `--learning_rate`: Learning rate (default: 2e-4)
- `--warmup_steps`: Warmup steps (default: 100)
- `--max_seq_length`: Maximum sequence length (default: 2048)

### Dataset Arguments

- `--dataset_name`: Dataset to use - `folio` or `proofwriter` (required)
- `--dataset_path`: Path to dataset JSON file (auto-detected if not provided)
- `--train_split`: Train/validation split ratio (default: 0.9)

### Output Arguments

- `--output_dir`: Output directory for model (default: ./lora_output)
- `--save_steps`: Save checkpoint every N steps (default: 100)
- `--logging_steps`: Log every N steps (default: 10)

## Dataset Format

Both FOLIO and ProofWriter datasets use the following JSON format:

```json
[
  {
    "system": "System prompt describing the task",
    "instruction": "The main problem/question with context",
    "input": "Additional input (e.g., 'The correct option is:')",
    "output": "The expected output (JSON format with reasoning and answer)"
  }
]
```

## Examples

See [finetune_examples.sh](finetune_examples.sh) for more examples, including:

- Different models (Llama, Gemma, Qwen, Mistral)
- Different quantization settings
- Custom LoRA parameters
- Custom target modules
- Longer training runs

## VRAM Requirements

Approximate VRAM requirements with 4-bit quantization:

- 7B models: ~6-8 GB VRAM
- 8B models: ~7-9 GB VRAM
- 13B models: ~10-12 GB VRAM

Without quantization or with 8-bit quantization, requirements will be higher.

## Using the Finetuned Model

After training, you can load and use the finetuned model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load LoRA weights
lora_model_path = "./lora_output/final_model"
model = PeftModel.from_pretrained(model, lora_model_path)

# Use the model
prompt = "### Instruction:\n[Your logical reasoning problem]\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Merging LoRA Weights

To merge LoRA weights into the base model for easier deployment:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# Load base model and LoRA
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = PeftModel.from_pretrained(base_model, "./lora_output/final_model")

# Merge and save
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged_model")
```

## Tips for Better Results

1. **Adjust batch size and gradient accumulation** based on your VRAM
2. **Experiment with LoRA rank**: Higher rank = more parameters but better capacity
3. **Learning rate**: 2e-4 to 3e-4 works well for most cases
4. **Training epochs**: 3-5 epochs is usually sufficient; more may lead to overfitting
5. **Use validation**: Monitor validation loss to prevent overfitting
6. **Target modules**: Including more modules (e.g., MLP layers) can improve performance

## Troubleshooting

### CUDA Out of Memory

- Reduce `--batch_size`
- Increase `--gradient_accumulation_steps`
- Use `--use_4bit` instead of `--use_8bit`
- Reduce `--max_seq_length`

### Model Not Found

- Make sure you're logged in to Hugging Face: `huggingface-cli login`
- Check that you have access to gated models (e.g., Llama requires approval)

### Slow Training

- Check that you're using GPU: `torch.cuda.is_available()`
- Ensure CUDA is properly installed
- Consider using a smaller model or fewer epochs

## License

This script is provided as-is for research and educational purposes. Please check the licenses of the base models and datasets you use.
