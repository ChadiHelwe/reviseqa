# GPU Device Selection Guide

Complete guide for selecting and using specific GPUs for training and evaluation.

## Quick Reference

### Use Specific GPU

```bash
# Method 1: Pass as argument
./train_all_models.sh folio 1              # Use GPU 1
./evaluate_all_models.sh data/ 2           # Use GPU 2
./run_lora_evaluation.sh 3                 # Use GPU 3

# Method 2: Set environment variable
export GPU_DEVICE=1
./train_all_models.sh folio

# Method 3: Inline environment variable
GPU_DEVICE=2 ./train_all_models.sh folio

# Method 4: Direct CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=1 python lora_finetune.py --model_name ...
```

### Use Multiple GPUs

```bash
# Use GPUs 0 and 1
./train_all_models.sh folio "0,1"

# Use GPUs 2 and 3
GPU_DEVICE="2,3" ./evaluate_all_models.sh data/

# Use all 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_finetune.py ...
```

## Checking Available GPUs

### List All GPUs
```bash
# Show all GPUs and their status
nvidia-smi

# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Show just GPU IDs and names
nvidia-smi --list-gpus

# Show GPU memory usage
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
```

### Find Free GPU
```bash
# Show memory usage for each GPU
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

# Find GPU with most free memory (simple script)
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t',' -k2 -rn | head -1 | cut -d',' -f1
```

## Usage Examples

### Training Examples

#### Single GPU Training

```bash
# Train on GPU 0 (default)
./train_all_models.sh folio

# Train on GPU 1
./train_all_models.sh folio 1

# Train on GPU 3
GPU_DEVICE=3 ./train_all_models.sh proofwriter

# Direct Python call with GPU 2
CUDA_VISIBLE_DEVICES=2 python lora_finetune.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --dataset_name folio \
  --output_dir ./lora_qwen2.5-7b_folio
```

#### Multi-GPU Training

```bash
# Use GPUs 0 and 1
./train_all_models.sh folio "0,1"

# Use GPUs 2 and 3
GPU_DEVICE="2,3" python lora_finetune.py \
  --model_name Qwen/Qwen2.5-14B-Instruct \
  --dataset_name folio \
  --output_dir ./lora_qwen2.5-14b_folio
```

**Note**: Multi-GPU training uses data parallelism automatically with PyTorch.

### Evaluation Examples

#### Single GPU Evaluation

```bash
# Evaluate on GPU 0 (default)
./run_lora_evaluation.sh

# Evaluate on GPU 1
./run_lora_evaluation.sh 1

# Evaluate all models on GPU 2
./evaluate_all_models.sh reviseqa_data/nl/verified-400/ 2

# Direct Python call with GPU 3
CUDA_VISIBLE_DEVICES=3 python lora_evaluation_complete.py \
  --data-dir reviseqa_data/nl/verified-400/ \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --lora-model ./lora_qwen2.5-7b_folio/final_model
```

#### Multi-GPU Evaluation

```bash
# Evaluate on GPUs 0 and 1 (parallel batches)
./run_lora_evaluation.sh "0,1"

# Evaluate all models on GPUs 2,3
GPU_DEVICE="2,3" ./evaluate_all_models.sh reviseqa_data/nl/verified-400/
```

## Advanced GPU Selection

### Automatically Select Free GPU

```bash
#!/bin/bash
# Auto-select GPU with most free memory
FREE_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t',' -k2 -rn | head -1 | cut -d',' -f1)

echo "Auto-selected GPU: $FREE_GPU"
GPU_DEVICE=$FREE_GPU ./train_all_models.sh folio
```

Save this as `auto_train.sh`:
```bash
#!/bin/bash
FREE_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t',' -k2 -rn | head -1 | cut -d',' -f1)
echo "Using GPU with most free memory: GPU $FREE_GPU"
export GPU_DEVICE=$FREE_GPU
./train_all_models.sh "$@"
```

### Run Different Models on Different GPUs

```bash
# Terminal 1: Train Qwen on GPU 0
CUDA_VISIBLE_DEVICES=0 python lora_finetune.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --dataset_name folio \
  --output_dir ./lora_qwen2.5-7b_folio &

# Terminal 2: Train Gemma on GPU 1
CUDA_VISIBLE_DEVICES=1 python lora_finetune.py \
  --model_name google/gemma-2-9b-it \
  --dataset_name folio \
  --output_dir ./lora_gemma2-9b_folio &

# Terminal 3: Train Llama on GPU 2
CUDA_VISIBLE_DEVICES=2 python lora_finetune.py \
  --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dataset_name folio \
  --output_dir ./lora_llama3.1-8b_folio &

# Wait for all to complete
wait
```

### Sequential Training on Single GPU

```bash
# Train multiple models sequentially on GPU 1
for MODEL in "Qwen/Qwen2.5-7B-Instruct" "google/gemma-2-9b-it"; do
  MODEL_NAME=$(echo $MODEL | sed 's/.*\///' | sed 's/-Instruct//')
  CUDA_VISIBLE_DEVICES=1 python lora_finetune.py \
    --model_name "$MODEL" \
    --dataset_name folio \
    --output_dir "./lora_${MODEL_NAME}_folio"
done
```

## GPU Memory Management

### Monitor Memory During Training

```bash
# Watch GPU memory in real-time (separate terminal)
watch -n 1 "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv"

# Or use gpustat (install with: pip install gpustat)
watch -n 1 gpustat --color
```

### Free GPU Memory

```bash
# Kill all Python processes (be careful!)
pkill -9 python

# Kill specific process by PID (find PID with nvidia-smi)
kill -9 <PID>

# Clear PyTorch cache (in Python)
import torch
torch.cuda.empty_cache()
```

### Optimize Memory Usage

```bash
# For limited VRAM, use smaller batch size
CUDA_VISIBLE_DEVICES=1 python lora_finetune.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --dataset_name folio \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --use_4bit

# For evaluation, reduce batch size
CUDA_VISIBLE_DEVICES=1 python lora_evaluation_complete.py \
  --data-dir reviseqa_data/nl/verified-400/ \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --lora-model ./lora_qwen2.5-7b_folio/final_model \
  --batch-size 1
```

## Multi-GPU Best Practices

### Data Parallel Training

PyTorch automatically uses data parallelism when multiple GPUs are specified:

```bash
# Train on GPUs 0,1,2,3
CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_finetune.py \
  --model_name Qwen/Qwen2.5-14B-Instruct \
  --dataset_name folio \
  --batch_size 2  # Per-GPU batch size
```

**Effective batch size** = batch_size × num_gpus × gradient_accumulation_steps
- Example: 2 × 4 GPUs × 4 = 32 effective batch size

### Load Balancing

When training multiple models simultaneously, balance by model size:

```bash
# GPUs 0-1: Large model (14B)
CUDA_VISIBLE_DEVICES=0,1 python lora_finetune.py \
  --model_name Qwen/Qwen2.5-14B-Instruct ... &

# GPU 2: Medium model (9B)
CUDA_VISIBLE_DEVICES=2 python lora_finetune.py \
  --model_name google/gemma-2-9b-it ... &

# GPU 3: Small model (7B)
CUDA_VISIBLE_DEVICES=3 python lora_finetune.py \
  --model_name Qwen/Qwen2.5-7B-Instruct ... &
```

## Troubleshooting

### "CUDA out of memory" Error

1. **Reduce batch size:**
   ```bash
   --batch_size 1 --gradient_accumulation_steps 16
   ```

2. **Use different GPU:**
   ```bash
   GPU_DEVICE=1 ./train_all_models.sh folio
   ```

3. **Clear GPU memory:**
   ```bash
   # Kill other processes using the GPU
   nvidia-smi  # Find PID
   kill -9 <PID>
   ```

4. **Use 4-bit quantization:**
   ```bash
   --use_4bit  # Should be default
   ```

### "No GPU available" Error

1. **Check GPU is visible:**
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Check CUDA installation:**
   ```bash
   nvcc --version
   python -c "import torch; print(torch.version.cuda)"
   ```

3. **Reinstall PyTorch with CUDA:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Wrong GPU Being Used

1. **Check CUDA_VISIBLE_DEVICES:**
   ```bash
   echo $CUDA_VISIBLE_DEVICES
   ```

2. **Override explicitly:**
   ```bash
   CUDA_VISIBLE_DEVICES=1 python lora_finetune.py ...
   ```

3. **Check for conflicts:**
   ```bash
   # Make sure no other env variable is set
   unset CUDA_DEVICE
   unset GPU_DEVICE
   ```

## Environment Variables Summary

| Variable | Purpose | Example |
|----------|---------|---------|
| `CUDA_VISIBLE_DEVICES` | Direct CUDA control | `CUDA_VISIBLE_DEVICES=1` |
| `GPU_DEVICE` | Script-level control | `GPU_DEVICE=1` |
| Both work, but `CUDA_VISIBLE_DEVICES` has priority | | |

## Complete Examples

### Train on Specific GPU
```bash
# Method 1: Argument (recommended for scripts)
./train_all_models.sh folio 1

# Method 2: Environment variable
export GPU_DEVICE=1
./train_all_models.sh folio

# Method 3: Inline
GPU_DEVICE=1 ./train_all_models.sh folio

# Method 4: Direct CUDA (for Python)
CUDA_VISIBLE_DEVICES=1 python lora_finetune.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --dataset_name folio
```

### Evaluate on Specific GPU
```bash
# Method 1: Argument
./run_lora_evaluation.sh 1

# Method 2: Environment variable
export GPU_DEVICE=2
./evaluate_all_models.sh reviseqa_data/nl/verified-400/

# Method 3: Direct CUDA
CUDA_VISIBLE_DEVICES=3 python lora_evaluation_complete.py \
  --data-dir reviseqa_data/nl/verified-400/ \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --lora-model ./lora_qwen2.5-7b_folio/final_model
```

### Multi-GPU Training
```bash
# Use GPUs 0,1,2,3
./train_all_models.sh folio "0,1,2,3"

# Or with environment variable
export GPU_DEVICE="0,1,2,3"
./train_all_models.sh folio
```

## Summary

✅ **Default behavior**: Uses GPU 0
✅ **Change GPU**: Pass as argument or set `GPU_DEVICE`
✅ **Multiple GPUs**: Use comma-separated list "0,1,2"
✅ **Direct control**: Use `CUDA_VISIBLE_DEVICES`
✅ **Check GPUs**: Use `nvidia-smi`

All scripts now support flexible GPU selection!
