#!/usr/bin/env python3
"""
Evaluation script for LoRA finetuned models
Compatible with the existing evaluation.py format and workflow
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm.auto import tqdm


class LoRAEvaluator:
    """Evaluator for LoRA finetuned models on logical reasoning tasks"""

    def __init__(self, args):
        self.args = args
        self.label_phrase = 'The correct option is:'

        # Load model
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """Load LoRA finetuned model"""
        print(f"Loading base model: {self.args.base_model}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.base_model,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Setup quantization if requested
        quantization_config = None
        if self.args.use_4bit:
            print("Using 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.args.use_8bit:
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
            self.args.base_model,
            **model_kwargs
        )

        # Load LoRA weights
        print(f"Loading LoRA weights from: {self.args.lora_model}")
        self.model = PeftModel.from_pretrained(base_model, self.args.lora_model)

        # Merge weights if requested
        if self.args.merge_weights:
            print("Merging LoRA weights into base model...")
            self.model = self.model.merge_and_unload()

        self.model.eval()
        print("Model loaded successfully")

    def load_in_context_examples(self):
        """Load in-context examples for few-shot prompting"""
        if self.args.dataset_name == 'ProverGen':
            with open(os.path.join(self.args.demonstration_path, f"{self.args.dataset_name}.json")) as f:
                example_dict = json.load(f)
            in_context_examples = example_dict[f"{self.args.split}_{self.args.mode}"]
        else:
            with open(os.path.join(self.args.demonstration_path, f'{self.args.dataset_name}.json'), 'r') as f:
                in_context_examples = json.load(f)
            in_context_examples = in_context_examples[self.args.mode]

        return in_context_examples

    def load_raw_dataset(self, split):
        """Load raw dataset from file"""
        with open(os.path.join(self.args.data_path, self.args.dataset_name, f'{split}.json')) as f:
            raw_dataset = json.load(f)

        return raw_dataset

    def create_prompt(self, in_context_example, test_example):
        """Create prompt from in-context examples and test example"""
        full_prompt = in_context_example
        context = test_example['context'].strip()
        question = test_example['question'].strip()
        options = '\n'.join([opt.strip() for opt in test_example['options']])
        full_prompt = full_prompt.replace('[[CONTEXT]]', context)
        full_prompt = full_prompt.replace('[[QUESTION]]', question)
        full_prompt = full_prompt.replace('[[OPTIONS]]', options)

        return full_prompt

    def format_messages(self, prompt_text):
        """Format prompt as messages for the model"""
        if self.args.mode == 'CoT':
            messages = [
                {'role': 'system', 'content': "Given a problem statement as contexts, the task is to answer a logical reasoning question. Your answer should be in JSON format with keys: reasoning, answer."},
                {'role': 'user', 'content': prompt_text}
            ]
        else:
            messages = [
                {'role': 'system', 'content': "Given a problem statement as contexts, the task is to answer a logical reasoning question. Your answer should be in JSON format with key: answer."},
                {'role': 'user', 'content': prompt_text}
            ]
        return messages

    def messages_to_prompt(self, messages):
        """Convert messages to a single prompt string"""
        # Check if tokenizer has chat template
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
            try:
                # Try to use the model's chat template
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return prompt
            except:
                pass

        # Fallback: manual formatting
        system_msg = messages[0]['content'] if messages[0]['role'] == 'system' else ""
        user_msg = messages[1]['content'] if len(messages) > 1 else messages[0]['content']

        # Simple template format
        if system_msg:
            prompt = f"### System:\n{system_msg}\n\n### User:\n{user_msg}\n\n### Assistant:\n"
        else:
            prompt = f"### User:\n{user_msg}\n\n### Assistant:\n"

        return prompt

    def completion(self, messages):
        """Generate completion for given messages"""
        # Convert messages to prompt
        prompt = self.messages_to_prompt(messages)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.args.max_seq_length
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.args.max_new_tokens,
                temperature=self.args.temperature if self.args.temperature > 0 else None,
                do_sample=self.args.temperature > 0,
                top_p=0.9 if self.args.temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated part
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response.strip()

    def evaluate(self):
        """Run evaluation on the dataset"""
        # Load raw dataset
        raw_dataset = self.load_raw_dataset(self.args.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.args.split} split.")

        # Load in-context examples
        if self.args.trained_model:
            in_context_examples = "Context:\n[[CONTEXT]]\n\nQuestion: [[QUESTION]]\n\nOptions:\n[[OPTIONS]]\n\nThe correct option is:"
        else:
            in_context_examples = self.load_in_context_examples()

        outputs = []
        cnt = -1
        for example in tqdm(raw_dataset):
            cnt += 1
            if cnt < self.args.start or cnt >= self.args.end:
                continue

            question = example['question']

            # Create prompt
            full_prompt = self.create_prompt(
                in_context_example=in_context_examples,
                test_example=example
            )

            # Format as messages
            messages = self.format_messages(full_prompt)

            # Get model response
            result = self.completion(messages)

            if self.args.verbose:
                print(f"\n{'='*80}")
                print(f"Example {cnt}")
                print(f"Prompt: {full_prompt[:200]}...")
                print(f"Response: {result}")
                print(f"{'='*80}\n")

            # Create output
            output = {
                'id': example['id'],
                'context': messages,  # Store the full prompt as messages
                'question': question,
                'label': example['answer'],
                'model_answer': result
            }
            outputs.append(output)

        # Prepare output filename
        model_name = self.args.lora_model.split('/')[-1]
        if not model_name:  # In case path ends with /
            model_name = self.args.lora_model.split('/')[-2]

        # Create output directory if it doesn't exist
        os.makedirs(self.args.output_dir, exist_ok=True)

        # Save outputs
        output_filename = f'{self.args.mode}_{self.args.dataset_name}_{self.args.split}_{model_name}_{self.args.start}-{self.args.end}.json'
        output_path = os.path.join(self.args.output_dir, output_filename)

        with open(output_path, 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {output_path}")

        # Compute and print accuracy
        self.compute_accuracy(outputs)

        return outputs

    def compute_accuracy(self, outputs):
        """Compute accuracy from outputs"""
        import re

        correct = 0
        total = 0

        for output in outputs:
            total += 1
            true_answer = output['label']
            model_answer_text = output['model_answer']

            # Extract answer from JSON or text
            predicted_answer = self.extract_answer(model_answer_text)

            if predicted_answer == true_answer:
                correct += 1

        accuracy = correct / total if total > 0 else 0

        print(f"\n{'='*80}")
        print(f"Evaluation Results")
        print(f"{'='*80}")
        print(f"Dataset: {self.args.dataset_name}")
        print(f"Split: {self.args.split}")
        print(f"Mode: {self.args.mode}")
        print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
        print(f"{'='*80}\n")

        return accuracy

    def extract_answer(self, text: str) -> str:
        """Extract answer from model output"""
        import re

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


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA finetuned models")

    # Model arguments
    parser.add_argument('--base_model', type=str, required=True,
                       help="Base model name or path")
    parser.add_argument('--lora_model', type=str, required=True,
                       help="Path to LoRA adapter weights")
    parser.add_argument('--use_4bit', action='store_true', default=True,
                       help="Use 4-bit quantization")
    parser.add_argument('--use_8bit', action='store_true',
                       help="Use 8-bit quantization")
    parser.add_argument('--no_quantization', action='store_true',
                       help="Disable quantization")
    parser.add_argument('--merge_weights', action='store_true',
                       help="Merge LoRA weights into base model")

    # Dataset arguments
    parser.add_argument('--dataset_name', type=str, default='FOLIO',
                       help="Dataset name (FOLIO, ProofWriter, ProntoQA, ProverGen)")
    parser.add_argument('--split', type=str, default='dev',
                       help="Dataset split (train, dev, test, etc.)")
    parser.add_argument('--mode', type=str, default='CoT',
                       help="Evaluation mode (Direct or CoT)")
    parser.add_argument('--data_path', type=str, default='Provergen/logic_data/',
                       help="Path to dataset directory")
    parser.add_argument('--demonstration_path', type=str, default='Provergen/logic_data/icl_examples',
                       help="Path to in-context learning examples")

    # Evaluation arguments
    parser.add_argument("--output_dir", type=str, default='lora_results/',
                       help="Output directory for results")
    parser.add_argument('--start', type=int, default=0,
                       help="Start index for evaluation")
    parser.add_argument('--end', type=int, default=None,
                       help="End index for evaluation (None for all)")
    parser.add_argument('--trained_model', action="store_true",
                       help="Use simple prompt format for finetuned models")

    # Generation arguments
    parser.add_argument('--temperature', type=float, default=0.0,
                       help="Sampling temperature")
    parser.add_argument('--max_new_tokens', type=int, default=None,
                       help="Maximum number of tokens to generate")
    parser.add_argument('--max_seq_length', type=int, default=2048,
                       help="Maximum input sequence length")
    parser.add_argument("--verbose", action="store_true",
                       help="Print verbose output")

    args = parser.parse_args()

    # Set max_new_tokens based on mode if not specified
    if args.max_new_tokens is None:
        if args.mode == "Direct":
            args.max_new_tokens = 128
        else:
            args.max_new_tokens = 1024

    # Handle quantization flags
    if args.no_quantization or args.merge_weights:
        args.use_4bit = False
        args.use_8bit = False
    elif args.use_8bit:
        args.use_4bit = False

    # Set end to a large number if not specified
    if args.end is None:
        args.end = 1000000  # Large number to process all examples

    print("="*80)
    print("LoRA Model Evaluation Configuration")
    print("="*80)
    print(f"Base Model: {args.base_model}")
    print(f"LoRA Model: {args.lora_model}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Split: {args.split}")
    print(f"Mode: {args.mode}")
    print(f"Range: {args.start}-{args.end}")
    print(f"Temperature: {args.temperature}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Quantization: {'4-bit' if args.use_4bit else '8-bit' if args.use_8bit else 'None'}")
    print("="*80 + "\n")

    # Initialize evaluator and run
    evaluator = LoRAEvaluator(args)
    evaluator.evaluate()


if __name__ == '__main__':
    main()
