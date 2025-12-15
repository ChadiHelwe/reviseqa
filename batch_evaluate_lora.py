#!/usr/bin/env python3
"""
Batch evaluation script for LoRA models across multiple datasets and modes
Automates running all evaluations similar to evaluation.py
"""

import argparse
import os
import subprocess
from pathlib import Path


def run_evaluation(
    base_model: str,
    lora_model: str,
    dataset_name: str,
    split: str,
    mode: str,
    output_dir: str,
    start: int = 0,
    end: int = None,
    temperature: float = 0.0,
    trained_model: bool = True,
    use_4bit: bool = True,
    verbose: bool = False,
):
    """Run a single evaluation"""
    cmd = [
        "python", "lora_evaluation.py",
        "--base_model", base_model,
        "--lora_model", lora_model,
        "--dataset_name", dataset_name,
        "--split", split,
        "--mode", mode,
        "--output_dir", output_dir,
        "--start", str(start),
        "--temperature", str(temperature),
    ]

    if end is not None:
        cmd.extend(["--end", str(end)])

    if trained_model:
        cmd.append("--trained_model")

    if use_4bit:
        cmd.append("--use_4bit")

    if verbose:
        cmd.append("--verbose")

    print(f"\n{'='*80}")
    print(f"Running evaluation:")
    print(f"  Dataset: {dataset_name} ({split})")
    print(f"  Mode: {mode}")
    print(f"  Model: {lora_model}")
    print(f"{'='*80}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"Warning: Evaluation failed for {dataset_name} {split} {mode}")

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation for LoRA models")

    # Model arguments
    parser.add_argument('--base_model', type=str, required=True,
                       help="Base model name or path")
    parser.add_argument('--lora_model', type=str, required=True,
                       help="Path to LoRA model")

    # Evaluation configuration
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['FOLIO', 'ProofWriter'],
                       help="Datasets to evaluate on")
    parser.add_argument('--splits', type=str, nargs='+',
                       default=['dev'],
                       help="Dataset splits to evaluate")
    parser.add_argument('--modes', type=str, nargs='+',
                       default=['CoT', 'Direct'],
                       help="Evaluation modes")

    # Optional arguments
    parser.add_argument('--output_dir', type=str, default='lora_results/',
                       help="Output directory")
    parser.add_argument('--temperature', type=float, default=0.0,
                       help="Sampling temperature")
    parser.add_argument('--start', type=int, default=0,
                       help="Start index")
    parser.add_argument('--end', type=int, default=None,
                       help="End index")
    parser.add_argument('--no_trained_model', action='store_true',
                       help="Don't use trained model prompt format")
    parser.add_argument('--no_4bit', action='store_true',
                       help="Don't use 4-bit quantization")
    parser.add_argument('--verbose', action='store_true',
                       help="Verbose output")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run evaluations
    results = []
    total = len(args.datasets) * len(args.splits) * len(args.modes)
    completed = 0

    print(f"\n{'='*80}")
    print(f"Batch Evaluation Configuration")
    print(f"{'='*80}")
    print(f"Base Model: {args.base_model}")
    print(f"LoRA Model: {args.lora_model}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Splits: {', '.join(args.splits)}")
    print(f"Modes: {', '.join(args.modes)}")
    print(f"Total evaluations: {total}")
    print(f"{'='*80}\n")

    for dataset in args.datasets:
        for split in args.splits:
            for mode in args.modes:
                success = run_evaluation(
                    base_model=args.base_model,
                    lora_model=args.lora_model,
                    dataset_name=dataset,
                    split=split,
                    mode=mode,
                    output_dir=args.output_dir,
                    start=args.start,
                    end=args.end,
                    temperature=args.temperature,
                    trained_model=not args.no_trained_model,
                    use_4bit=not args.no_4bit,
                    verbose=args.verbose,
                )

                completed += 1
                results.append({
                    'dataset': dataset,
                    'split': split,
                    'mode': mode,
                    'success': success,
                })

                print(f"\nProgress: {completed}/{total} evaluations completed")

    # Print summary
    print(f"\n{'='*80}")
    print(f"Evaluation Summary")
    print(f"{'='*80}")
    successful = sum(1 for r in results if r['success'])
    print(f"Successful: {successful}/{total}")
    print(f"Failed: {total - successful}/{total}")

    if total - successful > 0:
        print("\nFailed evaluations:")
        for r in results:
            if not r['success']:
                print(f"  - {r['dataset']} {r['split']} {r['mode']}")

    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
