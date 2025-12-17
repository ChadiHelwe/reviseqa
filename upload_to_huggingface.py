#!/usr/bin/env python3
"""
Upload trained LoRA models to Hugging Face Hub

This script uploads your fine-tuned LoRA models to Hugging Face Hub with proper
metadata, model cards, and organization.

Requirements:
    pip install huggingface-hub

Usage:
    # Login to Hugging Face (one-time)
    huggingface-cli login

    # Upload all models
    python upload_to_huggingface.py

    # Upload specific model
    python upload_to_huggingface.py --model lora_qwen2.5-7b_folio

    # Dry run (don't actually upload)
    python upload_to_huggingface.py --dry-run

    # Upload to organization instead of personal account
    python upload_to_huggingface.py --organization your-org-name
"""

import argparse
import os
import json
from pathlib import Path
from typing import Dict, List, Optional
from huggingface_hub import HfApi, create_repo, upload_folder, login


# Model metadata configuration
# Keys must match the directory names (after removing "lora_" prefix)
MODEL_METADATA = {
    "qwen2.5-7b_folio": {
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "dataset": "FOLIO",
        "description": "LoRA fine-tuned Qwen2.5-7B-Instruct on FOLIO logical reasoning dataset",
        "tags": ["lora", "qwen2.5", "logical-reasoning", "folio"],
        "hf_name": "qwen2.5-7b-folio",  # Hugging Face repo name (with dash)
    },
    "qwen2.5-7b_proofwriter": {
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "dataset": "ProofWriter",
        "description": "LoRA fine-tuned Qwen2.5-7B-Instruct on ProofWriter logical reasoning dataset",
        "tags": ["lora", "qwen2.5", "logical-reasoning", "proofwriter"],
        "hf_name": "qwen2.5-7b-proofwriter",
    },
    "qwen2.5-14b_folio": {
        "base_model": "Qwen/Qwen2.5-14B-Instruct",
        "dataset": "FOLIO",
        "description": "LoRA fine-tuned Qwen2.5-14B-Instruct on FOLIO logical reasoning dataset",
        "tags": ["lora", "qwen2.5", "logical-reasoning", "folio"],
        "hf_name": "qwen2.5-14b-folio",
    },
    "qwen2.5-14b_proofwriter": {
        "base_model": "Qwen/Qwen2.5-14B-Instruct",
        "dataset": "ProofWriter",
        "description": "LoRA fine-tuned Qwen2.5-14B-Instruct on ProofWriter logical reasoning dataset",
        "tags": ["lora", "qwen2.5", "logical-reasoning", "proofwriter"],
        "hf_name": "qwen2.5-14b-proofwriter",
    },
    "gemma2-9b_folio": {
        "base_model": "google/gemma-2-9b-it",
        "dataset": "FOLIO",
        "description": "LoRA fine-tuned Gemma-2-9B-IT on FOLIO logical reasoning dataset",
        "tags": ["lora", "gemma2", "logical-reasoning", "folio"],
        "hf_name": "gemma2-9b-folio",
    },
    "gemma2-9b_proofwriter": {
        "base_model": "google/gemma-2-9b-it",
        "dataset": "ProofWriter",
        "description": "LoRA fine-tuned Gemma-2-9B-IT on ProofWriter logical reasoning dataset",
        "tags": ["lora", "gemma2", "logical-reasoning", "proofwriter"],
        "hf_name": "gemma2-9b-proofwriter",
    },
    "gemma-7b_proofwriter": {
        "base_model": "google/gemma-7b-it",
        "dataset": "ProofWriter",
        "description": "LoRA fine-tuned Gemma-7B-IT on ProofWriter logical reasoning dataset",
        "tags": ["lora", "gemma", "logical-reasoning", "proofwriter"],
        "hf_name": "gemma-7b-proofwriter",
    },
}


def generate_model_card(model_name: str, metadata: Dict) -> str:
    """Generate a README.md model card for the model."""
    # Use hf_name for display if available
    display_name = metadata.get("hf_name", model_name)

    return f"""---
license: apache-2.0
base_model: {metadata['base_model']}
tags:
{chr(10).join(f'- {tag}' for tag in metadata['tags'])}
datasets:
- {metadata['dataset'].lower()}
language:
- en
library_name: peft
---

# {display_name.replace('-', ' ').replace('_', ' ').title()}

{metadata['description']}

## Model Details

- **Base Model**: [{metadata['base_model']}](https://huggingface.co/{metadata['base_model']})
- **Training Dataset**: {metadata['dataset']}
- **Training Method**: LoRA (Low-Rank Adaptation)
- **Framework**: PyTorch + Transformers + PEFT

## Training Details

This model was fine-tuned using LoRA on the {metadata['dataset']} dataset for logical reasoning tasks.

### Training Configuration

- **LoRA Rank**: 8 (typical)
- **LoRA Alpha**: 16 (typical)
- **Target Modules**: Attention layers
- **Quantization**: 4-bit precision during training

## Usage

To use this model, you need to install the required dependencies:

```bash
pip install transformers peft torch
```

### Loading the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "{metadata['base_model']}",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("{metadata['base_model']}")

# Load LoRA weights
model = PeftModel.from_pretrained(base_model, "YOUR_USERNAME/{metadata.get('hf_name', model_name)}")

# Generate text
prompt = "Your prompt here"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Merging LoRA Weights (Optional)

If you want to merge the LoRA weights with the base model:

```python
# Merge and save
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")
```

## Evaluation Results

This model was evaluated on the ReviseQA benchmark for logical reasoning.

For detailed evaluation results, please refer to the associated paper or repository.

## Intended Use

This model is intended for:
- Logical reasoning tasks
- Question answering with deductive reasoning
- Educational purposes
- Research in natural language understanding

## Limitations

- The model inherits limitations from the base model
- Performance may vary on out-of-distribution examples
- May require careful prompt engineering for optimal results

## Citation

If you use this model, please cite:

```bibtex
@misc{{{metadata.get('hf_name', model_name).replace('-', '_').replace('.', '_')},
  author = {{Your Name}},
  title = {{{metadata.get('hf_name', model_name)}}},
  year = {{2025}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/YOUR_USERNAME/{metadata.get('hf_name', model_name)}}}}},
}}
```

## License

This model is released under the Apache 2.0 license, consistent with the base model's license.
"""


def find_local_models() -> List[tuple]:
    """Find all local LoRA models by scanning directories."""
    models = []
    cwd = Path.cwd()

    # Find all directories starting with "lora_"
    for item in cwd.iterdir():
        if not item.is_dir() or not item.name.startswith("lora_"):
            continue

        # Skip result directories
        if "result" in item.name.lower():
            continue

        # Extract model name by removing "lora_" prefix
        model_name = item.name[5:]  # Remove "lora_" prefix

        # Check if this model has metadata
        if model_name not in MODEL_METADATA:
            print(f"⚠ Warning: No metadata for {model_name}, skipping")
            continue

        lora_dir = item
        final_model_dir = lora_dir / "final_model"

        # Always prefer final_model subdirectory if it exists
        if final_model_dir.exists() and any(final_model_dir.iterdir()):
            models.append((model_name, final_model_dir))
        elif lora_dir.exists() and any(lora_dir.iterdir()):
            models.append((model_name, lora_dir))

    return models


def upload_model(
    model_name: str,
    model_path: Path,
    username: str,
    organization: Optional[str] = None,
    private: bool = False,
    dry_run: bool = False
) -> bool:
    """Upload a model to Hugging Face Hub."""

    metadata = MODEL_METADATA.get(model_name)
    if not metadata:
        print(f"⚠ No metadata found for {model_name}, skipping")
        return False

    # Use hf_name if specified, otherwise use model_name
    hf_model_name = metadata.get("hf_name", model_name)

    # Determine repo name
    if organization:
        repo_id = f"{organization}/{hf_model_name}"
    else:
        repo_id = f"{username}/{hf_model_name}"

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Uploading {model_name} to {repo_id}")
    print(f"  Local path: {model_path}")
    print(f"  Base model: {metadata['base_model']}")
    print(f"  Dataset: {metadata['dataset']}")

    if dry_run:
        print("  ✓ Dry run - skipping actual upload")
        return True

    try:
        api = HfApi()

        # Create repository
        print(f"  Creating repository...")
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True
        )
        print(f"  ✓ Repository created/exists")

        # Generate and upload model card
        print(f"  Generating model card...")
        model_card = generate_model_card(model_name, metadata)
        readme_path = model_path / "README.md"

        # Temporarily create README if it doesn't exist
        readme_existed = readme_path.exists()
        with open(readme_path, 'w') as f:
            f.write(model_card)

        # Upload the entire folder
        print(f"  Uploading model files...")
        upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {model_name} LoRA weights"
        )

        # Clean up temporary README if we created it
        if not readme_existed:
            readme_path.unlink()

        print(f"  ✓ Successfully uploaded to https://huggingface.co/{repo_id}")
        return True

    except Exception as e:
        print(f"  ✗ Failed to upload {model_name}: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Upload LoRA models to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Specific model name to upload (e.g., qwen2.5-7b-folio). If not specified, uploads all models."
    )
    parser.add_argument(
        "--organization",
        type=str,
        help="Upload to organization instead of personal account"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the model repositories private"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading"
    )
    parser.add_argument(
        "--username",
        type=str,
        help="Hugging Face username (auto-detected if logged in)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Hugging Face Model Upload Script")
    print("=" * 80)

    # Check if logged in
    if not args.dry_run:
        try:
            api = HfApi()
            user_info = api.whoami()
            username = args.username or user_info['name']
            print(f"✓ Logged in as: {username}")
        except Exception as e:
            print(f"✗ Not logged in to Hugging Face!")
            print(f"  Please run: huggingface-cli login")
            return 1
    else:
        username = args.username or "YOUR_USERNAME"
        print(f"[DRY RUN MODE] - No actual uploads will be performed")

    print()

    # Find models to upload
    if args.model:
        model_name = args.model.replace("lora_", "")

        # Always try final_model first
        model_path = Path(f"lora_{model_name}/final_model")
        if not model_path.exists() or not any(model_path.iterdir()):
            model_path = Path(f"lora_{model_name}")

        if not model_path.exists() or not any(model_path.iterdir()):
            print(f"✗ Model not found: {model_path}")
            return 1

        models_to_upload = [(model_name, model_path)]
    else:
        models_to_upload = find_local_models()

    if not models_to_upload:
        print("✗ No models found to upload")
        print("\nAvailable models:")
        for name in MODEL_METADATA.keys():
            print(f"  - {name}")
        return 1

    print(f"Found {len(models_to_upload)} model(s) to upload:")
    for name, path in models_to_upload:
        print(f"  - {name} ({path})")

    if not args.dry_run:
        response = input("\nProceed with upload? [y/N]: ")
        if response.lower() != 'y':
            print("Upload cancelled")
            return 0

    print()

    # Upload models
    success_count = 0
    fail_count = 0

    for model_name, model_path in models_to_upload:
        if upload_model(
            model_name,
            model_path,
            username,
            args.organization,
            args.private,
            args.dry_run
        ):
            success_count += 1
        else:
            fail_count += 1

    # Summary
    print("\n" + "=" * 80)
    print("Upload Summary")
    print("=" * 80)
    print(f"✓ Successful: {success_count}")
    if fail_count > 0:
        print(f"✗ Failed: {fail_count}")
    print()

    if not args.dry_run and success_count > 0:
        print("Models uploaded successfully!")
        print("\nNext steps:")
        print("1. Visit your Hugging Face profile to view the models")
        print("2. Update model cards with additional information if needed")
        print("3. Share your models with the community!")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    exit(main())
