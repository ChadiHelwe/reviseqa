#!/usr/bin/env python3
"""
LoRA Finetuning Script for Logical Reasoning Models
Supports: Llama, Gemma, Qwen, and other causal language models
Datasets: FOLIO and ProofWriter
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from tqdm import tqdm


@dataclass
class FinetuneConfig:
    """Configuration for finetuning"""
    # Model settings
    model_name: str
    use_4bit: bool = True
    use_8bit: bool = False

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None

    # Training settings
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    max_seq_length: int = 2048

    # Dataset settings
    dataset_name: str = "folio"  # "folio" or "proofwriter"
    train_split: float = 0.9

    # Output settings
    output_dir: str = "./lora_output"
    save_steps: int = 100
    logging_steps: int = 10


class DatasetLoader:
    """Load and preprocess FOLIO and ProofWriter datasets"""

    def __init__(self, dataset_path: str, tokenizer, max_length: int = 2048):
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_json(self) -> List[Dict]:
        """Load dataset from JSON file"""
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def format_folio_prompt(self, example: Dict) -> str:
        """Format FOLIO dataset example into a prompt"""
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")

        # Combine instruction and input
        if input_text:
            prompt = f"{instruction}\n\n{input_text}"
        else:
            prompt = instruction

        return prompt

    def format_proofwriter_prompt(self, example: Dict) -> str:
        """Format ProofWriter dataset example into a prompt"""
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")

        # Combine instruction and input
        if input_text:
            prompt = f"{instruction}\n\n{input_text}"
        else:
            prompt = instruction

        return prompt

    def create_training_example(self, example: Dict, dataset_type: str) -> str:
        """Create a complete training example with prompt and response"""
        # Format the prompt based on dataset type
        if dataset_type == "folio":
            prompt = self.format_folio_prompt(example)
        elif dataset_type == "proofwriter":
            prompt = self.format_proofwriter_prompt(example)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        # Get the output/response
        output = example.get("output", "")

        # Combine into a training format
        # Using a simple instruction format
        full_text = f"### Instruction:\n{prompt}\n\n### Response:\n{output}"

        return full_text

    def tokenize_function(self, examples: Dict, dataset_type: str) -> Dict:
        """Tokenize the examples"""
        texts = []
        for i in range(len(examples.get("instruction", []))):
            example = {
                "instruction": examples["instruction"][i],
                "input": examples.get("input", [""] * len(examples["instruction"]))[i],
                "output": examples["output"][i],
                "system": examples.get("system", [""] * len(examples["instruction"]))[i],
            }
            text = self.create_training_example(example, dataset_type)
            texts.append(text)

        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None,
        )

        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    def load_and_prepare(self, dataset_type: str, train_split: float = 0.9) -> DatasetDict:
        """Load and prepare the dataset for training"""
        print(f"Loading {dataset_type} dataset from {self.dataset_path}...")

        # Load raw data
        raw_data = self.load_json()
        print(f"Loaded {len(raw_data)} examples")

        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list(raw_data)

        # Split into train and validation
        split_dataset = dataset.train_test_split(
            test_size=1.0 - train_split,
            seed=42
        )

        print(f"Train examples: {len(split_dataset['train'])}")
        print(f"Validation examples: {len(split_dataset['test'])}")

        # Tokenize
        print("Tokenizing dataset...")
        tokenized_dataset = split_dataset.map(
            lambda examples: self.tokenize_function(examples, dataset_type),
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing",
        )

        return tokenized_dataset


class LoRAFinetuner:
    """LoRA Finetuner for causal language models"""

    def __init__(self, config: FinetuneConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.dataset = None

    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer with quantization if specified"""
        print(f"Loading model: {self.config.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )

        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Setup quantization config
        quantization_config = None
        if self.config.use_4bit:
            print("Using 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.config.use_8bit:
            print("Using 8-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
        }

        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )

        # Prepare model for k-bit training if quantized
        if quantization_config is not None:
            self.model = prepare_model_for_kbit_training(self.model)

        print("Model and tokenizer loaded successfully")

    def get_target_modules(self) -> List[str]:
        """Get target modules for LoRA based on model architecture"""
        if self.config.target_modules is not None:
            return self.config.target_modules

        # Auto-detect based on model type
        model_name_lower = self.config.model_name.lower()

        if "llama" in model_name_lower:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "gemma" in model_name_lower:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "qwen" in model_name_lower:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "mistral" in model_name_lower:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            # Default: target attention layers
            print("Warning: Unknown model type, using default target modules")
            return ["q_proj", "v_proj"]

    def setup_lora(self):
        """Setup LoRA configuration and apply to model"""
        print("Setting up LoRA...")

        target_modules = self.get_target_modules()
        print(f"Target modules: {target_modules}")

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        print("LoRA setup complete")

    def load_dataset(self, dataset_path: str):
        """Load and prepare dataset"""
        loader = DatasetLoader(
            dataset_path=dataset_path,
            tokenizer=self.tokenizer,
            max_length=self.config.max_seq_length,
        )

        self.dataset = loader.load_and_prepare(
            dataset_type=self.config.dataset_name,
            train_split=self.config.train_split,
        )

    def train(self):
        """Train the model with LoRA"""
        print("Starting training...")

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.save_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            fp16=True,
            optim="paged_adamw_8bit" if self.config.use_4bit or self.config.use_8bit else "adamw_torch",
            report_to="none",  # Change to "wandb" if you want to use Weights & Biases
            save_total_limit=3,
            gradient_checkpointing=True,
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
        )

        # Train
        trainer.train()

        # Save final model
        final_model_path = os.path.join(self.config.output_dir, "final_model")
        trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)

        print(f"Training complete! Model saved to {final_model_path}")

        return trainer


def main():
    parser = argparse.ArgumentParser(description="LoRA Finetune models on FOLIO and ProofWriter datasets")

    # Model arguments
    parser.add_argument("--model_name", type=str, required=True,
                       help="Model name or path (e.g., meta-llama/Llama-2-7b-hf, google/gemma-7b, Qwen/Qwen2-7B)")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                       help="Use 4-bit quantization")
    parser.add_argument("--use_8bit", action="store_true",
                       help="Use 8-bit quantization")
    parser.add_argument("--no_quantization", action="store_true",
                       help="Disable quantization")

    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                       help="LoRA dropout")
    parser.add_argument("--target_modules", type=str, nargs="+", default=None,
                       help="Target modules for LoRA")

    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Warmup steps")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                       help="Maximum sequence length")

    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, choices=["folio", "proofwriter"], required=True,
                       help="Dataset to use")
    parser.add_argument("--dataset_path", type=str, default=None,
                       help="Path to dataset JSON file (auto-detected if not provided)")
    parser.add_argument("--train_split", type=float, default=0.9,
                       help="Train/validation split ratio")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./lora_output",
                       help="Output directory for model")
    parser.add_argument("--save_steps", type=int, default=100,
                       help="Save checkpoint every N steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Log every N steps")

    args = parser.parse_args()

    # Auto-detect dataset path if not provided
    if args.dataset_path is None:
        base_dir = Path(__file__).parent
        if args.dataset_name == "folio":
            args.dataset_path = base_dir / "Provergen/training_data/folio.json"
        elif args.dataset_name == "proofwriter":
            args.dataset_path = base_dir / "Provergen/training_data/proofwriter-5000.json"

        print(f"Using auto-detected dataset path: {args.dataset_path}")

    # Check if dataset exists
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset file not found at {args.dataset_path}")
        sys.exit(1)

    # Handle quantization flags
    if args.no_quantization:
        args.use_4bit = False
        args.use_8bit = False
    elif args.use_8bit:
        args.use_4bit = False

    # Create config
    config = FinetuneConfig(
        model_name=args.model_name,
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_seq_length=args.max_seq_length,
        dataset_name=args.dataset_name,
        train_split=args.train_split,
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
    )

    print("=" * 80)
    print("LoRA Finetuning Configuration")
    print("=" * 80)
    print(f"Model: {config.model_name}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Output directory: {config.output_dir}")
    print(f"Quantization: {'4-bit' if config.use_4bit else '8-bit' if config.use_8bit else 'None'}")
    print(f"LoRA rank: {config.lora_r}, alpha: {config.lora_alpha}, dropout: {config.lora_dropout}")
    print(f"Epochs: {config.num_epochs}, Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print("=" * 80)

    # Initialize finetuner
    finetuner = LoRAFinetuner(config)

    # Setup model and tokenizer
    finetuner.setup_model_and_tokenizer()

    # Setup LoRA
    finetuner.setup_lora()

    # Load dataset
    finetuner.load_dataset(args.dataset_path)

    # Train
    finetuner.train()

    print("\nFinetuning completed successfully!")


if __name__ == "__main__":
    main()
