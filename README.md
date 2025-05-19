# ReviseQA

ReviseQA is a comprehensive framework for generating and evaluating logical reasoning datasets that combine First-Order Logic (FOL) and natural language processing. The system creates challenging reasoning problems by systematically modifying logical assumptions and evaluating model performance on the resulting datasets.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Generating Datasets](#generating-datasets)
  - [Running Evaluations](#running-evaluations)
  - [Verifying Datasets](#verifying-datasets)
- [Project Structure](#project-structure)
- [Dataset Format](#dataset-format)
- [Models Supported](#models-supported)
- [Contributing](#contributing)
- [Citation](#citation)

## Overview

ReviseQA addresses the challenge of evaluating language models' logical reasoning capabilities by:
- Converting ProverGen datasets to FOL format
- Systematically modifying logical assumptions
- Generating natural language versions of logical problems
- Evaluating model performance across different reasoning configurations

The system uses automated theorem proving (Prover9) to verify logical consistency and provides detailed metrics on model performance.

## Features

- **Dataset Generation**: Convert between FOL and natural language representations
- **Systematic Modifications**: Generate variations by adding/removing facts and rules
- **Multi-Model Evaluation**: Support for various LLMs including Claude, GPT-4, Gemini, etc.
- **Verification Tools**: Ensure dataset consistency and validity
- **Performance Metrics**: Detailed analysis with confidence intervals and degradation tracking
- **Parallel Processing**: Efficient generation and evaluation using multi-threading

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/reviseqa.git
cd reviseqa
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install theorem prover (required for logical verification):
```bash
# Ubuntu/Debian
sudo apt-get install prover9

# macOS
brew install prover9
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys for the models you want to use
```

N.B: To generate the initial examples, please check the ProverGen repo [here](https://github.com/opendatalab/ProverGen), in addition if you find issues while compiling prover9 check the README file in the ProverGen repo [here](https://github.com/opendatalab/ProverGen/blob/main/README.md).


## Quick Start

1. Save your OPENROUTER api key in a ```.env``` file

2. Generate a FOL dataset from ProverGen data:
```bash
python run.py generate-fol --input-file provergen_data/translated_data/easy-10-0_10.json
```

3. Verify the generated FOL dataset:
```bash
python run.py verify --verify-type fol
```

4. Convert to natural language:
```bash
python run.py generate-nl --input-dir reviseqa_data/verification_1_fol
```

5. Run evaluation:
```bash
python run.py evaluate --data-dir reviseqa_data/nl/verified --model-name google/gemini-2.5-flash-preview
```

## Usage

### Generating Datasets

#### FOL Dataset Generation
Generate FOL datasets from ProverGen data:
```bash
python run.py generate-fol --input-file <provergen_file.json> [--parallel]
```

Options:
- `--input-file`: Path to ProverGen JSON file
- `--parallel`: Use parallel processing (recommended for large datasets)

#### Natural Language Generation
Convert FOL datasets to natural language:
```bash
python run.py generate-nl --input-dir <fol_directory>
```

Options:
- `--input-dir`: Directory containing FOL JSON files

### Running Evaluations

Evaluate models on generated datasets:
```bash
python run.py evaluate \
    --data-dir <dataset_directory> \
    --model-name <model_identifier> \
    --results-dir results \
    [--guided] \
    [--batch-size 32] \
    [--enable-truncated] \
    [--enable-shuffled]
```

Options:
- `--data-dir`: Directory containing evaluation datasets
- `--model-name`: Model identifier (e.g., `anthropic/claude-3.7-sonnet`)
- `--results-dir`: Output directory for results
- `--guided`: Enable structured output mode
- `--batch-size`: Number of parallel workers
- `--enable-truncated`: Use truncated reasoning
- `--enable-shuffled`: Use shuffled datasets

### Verifying Datasets

Verify FOL dataset consistency:
```bash
# Check FOL consistency
python run.py verify --verify-type fol


## Project Structure

```
reviseqa/
├── run.py                  # Main entry point
├── src/
│   ├── evaluation.py       # Model evaluation logic
│   ├── generate_reviseqa.py    # FOL dataset generation
│   ├── generate_reviseqa_nl.py # Natural language generation
│   ├── prompt_engine.py    # Prompt templates and management
│   ├── prover.py          # Theorem proving integration
│   ├── data_structure.py  # Core data structures
│   ├── utils.py           # Utility functions
│   └──  verification_fol.py  # FOL verification
├── scripts/
│   └── evaluate_all.sh    # Batch evaluation script
├── data/                  # Static data files
├── reviseqa_data/        # Generated datasets
├── provergen_data/       # Input data from ProverGen
└── results/              # Evaluation results
```

## Dataset Format

Each example contains:
- `original_context`: List of natural language premises
- `original_context_fol`: FOL representation of premises
- `conclusion`: Statement to evaluate
- `conclusion_fol`: FOL representation of conclusion
- `answer`: Ground truth (True/False/Uncertain)
- `reasoning_chain`: Step-by-step reasoning
- `edits_made`: Modifications applied during generation

## Models Supported

ReviseQA supports evaluation with various language models:

- **Anthropic**: Claude models (e.g., `anthropic/claude-3.7-sonnet`)
- **OpenAI**: GPT models (e.g., `openai/gpt-4.1-mini`)
- **Google**: Gemini models (e.g., `google/gemini-2.5-flash-preview`)
- **Mistral**: Mistral models (e.g., `mistral/ministral-8b`)
- **Others**: Any model accessible via compatible APIs

### Model Naming Convention
Models are specified as `provider/model-name`, optionally with `:thinking` suffix for thinking-enabled variants.

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Run verification (`python run.py verify --verify-type fol`)
5. Submit a pull request

## Citation

If you use ReviseQA in your research, please cite:

```bibtex
@article{reviseqa2024,
  title={ReviseQA: Evaluating Logical Reasoning through Systematic Modification},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on top of the ProverGen dataset
- Uses Prover9 for automated theorem proving
- Inspired by research in logical reasoning evaluation