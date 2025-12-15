#!/usr/bin/env python3
"""
Evaluate LoRA finetuned model on test dataset
Compute accuracy and other metrics
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm


class ModelEvaluator:
    """Evaluate finetuned models on logical reasoning datasets"""

    def __init__(
        self,
        base_model_name: str,
        lora_model_path: str,
        use_4bit: bool = True,
    ):
        self.base_model_name = base_model_name
        self.lora_model_path = lora_model_path
        self.use_4bit = use_4bit

        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """Load model and tokenizer"""
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
        if self.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
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
        self.model.eval()

        print("Model loaded successfully")

    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        """Format prompt"""
        if input_text:
            prompt = f"{instruction}\n\n{input_text}"
        else:
            prompt = instruction

        return f"### Instruction:\n{prompt}\n\n### Response:\n"

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate response"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,  # Low temperature for more deterministic output
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip()

    def extract_answer(self, text: str) -> str:
        """Extract answer from model output"""
        # Try to parse JSON
        try:
            # Find JSON in the text
            json_match = re.search(r'\{[^}]*"answer"\s*:\s*"([^"]*)"[^}]*\}', text)
            if json_match:
                return json_match.group(1)

            # Try direct JSON parse
            data = json.loads(text)
            if "answer" in data:
                return data["answer"]
        except:
            pass

        # Fallback: look for "answer": "X" pattern
        match = re.search(r'"answer"\s*:\s*"([^"]*)"', text)
        if match:
            return match.group(1)

        # Fallback: look for A, B, or C
        match = re.search(r'\b([ABC])\b', text)
        if match:
            return match.group(1)

        return ""

    def evaluate_dataset(
        self,
        dataset_path: str,
        max_examples: int = None,
    ) -> Dict:
        """Evaluate on a dataset"""
        print(f"Loading dataset from {dataset_path}")

        with open(dataset_path, 'r') as f:
            data = json.load(f)

        if max_examples:
            data = data[:max_examples]

        print(f"Evaluating on {len(data)} examples...")

        correct = 0
        total = 0
        results = []

        for example in tqdm(data, desc="Evaluating"):
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            ground_truth_output = example.get("output", "")

            # Generate prediction
            prompt = self.format_prompt(instruction, input_text)
            prediction = self.generate(prompt)

            # Extract answers
            predicted_answer = self.extract_answer(prediction)
            true_answer = self.extract_answer(ground_truth_output)

            # Check if correct
            is_correct = predicted_answer == true_answer
            if is_correct:
                correct += 1
            total += 1

            # Store result
            results.append({
                "instruction": instruction,
                "input": input_text,
                "prediction": prediction,
                "predicted_answer": predicted_answer,
                "ground_truth": ground_truth_output,
                "true_answer": true_answer,
                "correct": is_correct,
            })

        # Compute metrics
        accuracy = correct / total if total > 0 else 0

        metrics = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }

        return metrics, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA finetuned model")

    parser.add_argument("--base_model", type=str, required=True,
                       help="Base model name or path")
    parser.add_argument("--lora_model", type=str, required=True,
                       help="Path to LoRA model")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to evaluation dataset")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json",
                       help="Output file for results")
    parser.add_argument("--max_examples", type=int, default=None,
                       help="Maximum number of examples to evaluate")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                       help="Use 4-bit quantization")

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = ModelEvaluator(
        base_model_name=args.base_model,
        lora_model_path=args.lora_model,
        use_4bit=args.use_4bit,
    )

    # Evaluate
    metrics, results = evaluator.evaluate_dataset(
        dataset_path=args.dataset_path,
        max_examples=args.max_examples,
    )

    # Print metrics
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    print("=" * 80)

    # Save results
    output = {
        "metrics": metrics,
        "results": results,
    }

    print(f"\nSaving results to {args.output_file}")
    with open(args.output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print("Evaluation complete!")


if __name__ == "__main__":
    main()
