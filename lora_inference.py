#!/usr/bin/env python3
"""
Inference script for LoRA finetuned models
Supports interactive and batch inference
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


class LoRAInference:
    """Inference engine for LoRA finetuned models"""

    def __init__(
        self,
        base_model_name: str,
        lora_model_path: str,
        use_4bit: bool = True,
        use_8bit: bool = False,
        merge_weights: bool = False,
    ):
        self.base_model_name = base_model_name
        self.lora_model_path = lora_model_path
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit
        self.merge_weights = merge_weights

        self.model = None
        self.tokenizer = None

        self.load_model()

    def load_model(self):
        """Load the model and tokenizer"""
        print(f"Loading base model: {self.base_model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Setup quantization
        quantization_config = None
        if not self.merge_weights:  # Only use quantization if not merging
            if self.use_4bit:
                print("Using 4-bit quantization")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            elif self.use_8bit:
                print("Using 8-bit quantization")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )

        # Load base model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
        }

        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            **model_kwargs
        )

        # Load LoRA weights
        print(f"Loading LoRA weights from: {self.lora_model_path}")
        self.model = PeftModel.from_pretrained(base_model, self.lora_model_path)

        # Merge weights if requested
        if self.merge_weights:
            print("Merging LoRA weights into base model...")
            self.model = self.model.merge_and_unload()

        self.model.eval()
        print("Model loaded successfully")

    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        """Format the prompt in the same way as training"""
        if input_text:
            prompt = f"{instruction}\n\n{input_text}"
        else:
            prompt = instruction

        full_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        return full_prompt

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
    ) -> str:
        """Generate response for a prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

        # Move to same device as model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated part
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response.strip()

    def infer_single(
        self,
        instruction: str,
        input_text: str = "",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Run inference on a single example"""
        prompt = self.format_prompt(instruction, input_text)
        response = self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return response

    def infer_batch(
        self,
        examples: List[Dict],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> List[str]:
        """Run inference on a batch of examples"""
        responses = []

        for example in examples:
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")

            response = self.infer_single(
                instruction=instruction,
                input_text=input_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            responses.append(response)

        return responses

    def infer_from_file(
        self,
        input_file: str,
        output_file: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ):
        """Run inference on examples from a JSON file"""
        print(f"Loading examples from {input_file}")

        with open(input_file, 'r') as f:
            examples = json.load(f)

        print(f"Running inference on {len(examples)} examples...")

        results = []
        for i, example in enumerate(examples):
            print(f"Processing {i+1}/{len(examples)}")

            instruction = example.get("instruction", "")
            input_text = example.get("input", "")

            response = self.infer_single(
                instruction=instruction,
                input_text=input_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

            result = {
                "instruction": instruction,
                "input": input_text,
                "output": response,
                "ground_truth": example.get("output", ""),
            }
            results.append(result)

        # Save results
        print(f"Saving results to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print("Inference complete!")

    def interactive_mode(self):
        """Run interactive inference"""
        print("\n" + "=" * 80)
        print("Interactive Inference Mode")
        print("Enter your logical reasoning problem (or 'quit' to exit)")
        print("=" * 80 + "\n")

        while True:
            print("\nEnter instruction:")
            instruction = input("> ")

            if instruction.lower() in ['quit', 'exit', 'q']:
                break

            print("\nEnter additional input (press Enter to skip):")
            input_text = input("> ")

            print("\nGenerating response...")
            response = self.infer_single(instruction, input_text)

            print("\n" + "-" * 80)
            print("Response:")
            print(response)
            print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Inference with LoRA finetuned models")

    # Model arguments
    parser.add_argument("--base_model", type=str, required=True,
                       help="Base model name or path")
    parser.add_argument("--lora_model", type=str, required=True,
                       help="Path to LoRA model")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                       help="Use 4-bit quantization")
    parser.add_argument("--use_8bit", action="store_true",
                       help="Use 8-bit quantization")
    parser.add_argument("--no_quantization", action="store_true",
                       help="Disable quantization")
    parser.add_argument("--merge_weights", action="store_true",
                       help="Merge LoRA weights into base model")

    # Inference mode
    parser.add_argument("--mode", type=str, choices=["interactive", "file", "single"],
                       default="interactive",
                       help="Inference mode")

    # Single inference arguments
    parser.add_argument("--instruction", type=str,
                       help="Instruction for single inference")
    parser.add_argument("--input", type=str, default="",
                       help="Input for single inference")

    # File inference arguments
    parser.add_argument("--input_file", type=str,
                       help="Input JSON file for batch inference")
    parser.add_argument("--output_file", type=str,
                       help="Output JSON file for batch inference")

    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=256,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")

    args = parser.parse_args()

    # Handle quantization flags
    if args.no_quantization or args.merge_weights:
        args.use_4bit = False
        args.use_8bit = False
    elif args.use_8bit:
        args.use_4bit = False

    # Initialize inference engine
    print("Initializing inference engine...")
    engine = LoRAInference(
        base_model_name=args.base_model,
        lora_model_path=args.lora_model,
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit,
        merge_weights=args.merge_weights,
    )

    # Run inference based on mode
    if args.mode == "interactive":
        engine.interactive_mode()

    elif args.mode == "single":
        if not args.instruction:
            print("Error: --instruction is required for single inference mode")
            return

        response = engine.infer_single(
            instruction=args.instruction,
            input_text=args.input,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        print("\n" + "=" * 80)
        print("Response:")
        print(response)
        print("=" * 80)

    elif args.mode == "file":
        if not args.input_file or not args.output_file:
            print("Error: --input_file and --output_file are required for file mode")
            return

        engine.infer_from_file(
            input_file=args.input_file,
            output_file=args.output_file,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )


if __name__ == "__main__":
    main()
