#!/usr/bin/env python3
"""
ReviseQA - Main Entry Point
A system for generating and evaluating first-order logic reasoning datasets
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def generate_fol_dataset(args):
    """Generate FOL dataset from ProverGen data"""
    from src.generate_reviseqa import make_dataset, parallel_make_dataset
    
    if args.parallel:
        parallel_make_dataset(args.input_file)
    else:
        make_dataset(args.input_file)


def generate_nl_dataset(args):
    """Generate natural language dataset from FOL data"""
    from src.generate_reviseqa_nl import parallel_make_dataset_nl
    
    parallel_make_dataset_nl(args.input_dir)


def evaluate_models(args):
    """Run evaluation on generated datasets"""
    from src.evaluation import main as evaluate_main
    
    eval_args = [
        '--data-dir', args.data_dir,
        '--results-dir', args.results_dir,
        '--batch-size', str(args.batch_size),
        '--model-name', args.model_name
    ]
    
    if args.guided:
        eval_args.append('--guided')
    if args.enable_truncated:
        eval_args.append('--enable_truncated')
    if args.enable_shuffled:
        eval_args.append('--enable_shuffled')
    
    # Call evaluation main with parsed args
    sys.argv = ['evaluation.py'] + eval_args
    evaluate_main()


def verify_datasets(args):
    """Run verification on datasets"""
    if args.verify_type == 'fol':
        from src.verification_fol import main as verify_fol_main
        verify_fol_main()
    else:
        print(f"Unknown verification type: {args.verify_type}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='ReviseQA: Generate and evaluate FOL reasoning datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate FOL dataset from ProverGen data
  python run.py generate-fol --input-file provergen_data/translated_data/hard-500-0_500.json

  # Verify dataset consistency
  python run.py verify --verify-type fol

  # Generate natural language dataset from FOL
  python run.py generate-nl --input-dir reviseqa_data/verification_1_fol

  # Run evaluation on a dataset
  python run.py evaluate --data-dir reviseqa_data/nl/verified --model-name anthropic/claude-3.7-sonnet

        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate FOL dataset
    fol_parser = subparsers.add_parser('generate-fol', help='Generate FOL dataset from ProverGen data')
    fol_parser.add_argument('--input-file', required=True, help='Input ProverGen JSON file')
    fol_parser.add_argument('--parallel', action='store_true', help='Use parallel processing')
    
    # Generate NL dataset
    nl_parser = subparsers.add_parser('generate-nl', help='Generate natural language dataset from FOL')
    nl_parser.add_argument('--input-dir', required=True, help='Input directory with FOL files')
    
    # Evaluate models
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate models on datasets')
    eval_parser.add_argument('--data-dir', required=True, help='Directory containing JSON example files')
    eval_parser.add_argument('--results-dir', default='results', help='Directory for saving metrics')
    eval_parser.add_argument('--batch-size', type=int, default=32, help='Number of parallel workers')
    eval_parser.add_argument('--model-name', default='google/gemini-2.5-flash-preview', 
                           help='LLM model identifier (format: provider/model)')
    eval_parser.add_argument('--guided', action='store_true', help='Enable structured-output guided mode')
    eval_parser.add_argument('--enable-truncated', action='store_true', help='Use truncated reasoning')
    eval_parser.add_argument('--enable-shuffled', action='store_true', help='Use shuffled datasets')
    
    # Verify datasets
    verify_parser = subparsers.add_parser('verify', help='Verify dataset consistency')
    verify_parser.add_argument('--verify-type', choices=['fol'], required=True,
                             help='Type of verification to run')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute the appropriate command
    if args.command == 'generate-fol':
        generate_fol_dataset(args)
    elif args.command == 'generate-nl':
        generate_nl_dataset(args)
    elif args.command == 'evaluate':
        evaluate_models(args)
    elif args.command == 'verify':
        verify_datasets(args)


if __name__ == '__main__':
    main()