#!/usr/bin/env python3
"""
Test script to download and run inference with uploaded LoRA models
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse


def test_model(model_id: str, base_model: str, test_prompt: str = None):
    """
    Download and test a LoRA model from Hugging Face

    Args:
        model_id: HuggingFace model ID (e.g., "Chadi1992/qwen2.5-7b-folio")
        base_model: Base model name (e.g., "Qwen/Qwen2.5-7B-Instruct")
        test_prompt: Optional custom test prompt
    """
    print("=" * 80)
    print(f"Testing Model: {model_id}")
    print("=" * 80)

    # Default test prompt for logical reasoning
    if test_prompt is None:
        test_prompt = """Given the following premises:
1. All humans are mortal.
2. Socrates is a human.

Question: Is Socrates mortal?

Answer:"""

    print(f"\nTest Prompt:\n{test_prompt}\n")
    print("-" * 80)

    try:
        # Load base model
        print(f"\n[1/4] Loading base model: {base_model}")
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("✓ Base model loaded successfully")

        # Load tokenizer
        print(f"\n[2/4] Loading tokenizer from: {base_model}")
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        print("✓ Tokenizer loaded successfully")

        # Load LoRA weights
        print(f"\n[3/4] Loading LoRA adapter from: {model_id}")
        model = PeftModel.from_pretrained(base_model_obj, model_id)
        print("✓ LoRA adapter loaded successfully")

        # Run inference
        print(f"\n[4/4] Running inference...")
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("\n" + "=" * 80)
        print("MODEL OUTPUT:")
        print("=" * 80)
        print(response)
        print("=" * 80)

        # Extract just the answer part
        if test_prompt in response:
            answer = response.split(test_prompt)[1].strip()
            print("\nExtracted Answer:")
            print(answer)

        print("\n✓ Test completed successfully!")
        return True

    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"   {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test downloaded LoRA models")
    parser.add_argument(
        "--model",
        type=str,
        default="Chadi1992/qwen2.5-7b-folio",
        help="HuggingFace model ID to test"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        help="Base model name (auto-detected if not provided)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Custom test prompt"
    )

    args = parser.parse_args()

    # Auto-detect base model from model ID
    model_to_base = {
        "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
        "gemma2-9b": "google/gemma-2-9b-it",
        "gemma-7b": "google/gemma-7b-it",
    }

    base_model = args.base_model
    if not base_model:
        for key, value in model_to_base.items():
            if key in args.model.lower():
                base_model = value
                break

    if not base_model:
        print("✗ Could not auto-detect base model. Please specify --base-model")
        return 1

    print(f"Using base model: {base_model}")

    success = test_model(args.model, base_model, args.prompt)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
