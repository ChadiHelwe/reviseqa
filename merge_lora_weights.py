#!/usr/bin/env python3
"""
Merge LoRA weights into base model for easier deployment
"""

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora_weights(
    base_model_name: str,
    lora_model_path: str,
    output_path: str,
    push_to_hub: bool = False,
    hub_model_id: Optional[str] = None,
):
    """Merge LoRA weights into base model"""
    print(f"Loading base model: {base_model_name}")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )

    # Load LoRA model
    print(f"Loading LoRA weights from: {lora_model_path}")
    model = PeftModel.from_pretrained(base_model, lora_model_path)

    # Merge weights
    print("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()

    # Save merged model
    print(f"Saving merged model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print("Merge complete!")

    # Push to hub if requested
    if push_to_hub:
        if not hub_model_id:
            raise ValueError("--hub_model_id is required when --push_to_hub is set")

        print(f"Pushing to Hugging Face Hub: {hub_model_id}")
        merged_model.push_to_hub(hub_model_id)
        tokenizer.push_to_hub(hub_model_id)
        print("Push complete!")

    # Print model info
    print("\nMerged model information:")
    print(f"  - Model saved to: {output_path}")
    print(f"  - Size: {sum(p.numel() for p in merged_model.parameters()) / 1e9:.2f}B parameters")

    return merged_model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base model")

    parser.add_argument("--base_model", type=str, required=True,
                       help="Base model name or path")
    parser.add_argument("--lora_model", type=str, required=True,
                       help="Path to LoRA model")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Output path for merged model")
    parser.add_argument("--push_to_hub", action="store_true",
                       help="Push merged model to Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str,
                       help="Model ID for Hugging Face Hub")

    args = parser.parse_args()

    merge_lora_weights(
        base_model_name=args.base_model,
        lora_model_path=args.lora_model,
        output_path=args.output_path,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )


if __name__ == "__main__":
    from typing import Optional
    main()
