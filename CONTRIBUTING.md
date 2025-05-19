# Contributing to ReviseQA

We welcome contributions to ReviseQA! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature or bug fix
4. Make your changes
5. Test your changes thoroughly
6. Submit a pull request

## Development Setup

1. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up pre-commit hooks (if available):
   ```bash
   pre-commit install
   ```

## Code Style

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Add docstrings to all functions and classes
- Keep functions focused and modular

## Testing

Before submitting a pull request:

1. Verify your changes work with existing datasets:
   ```bash
   python run.py verify --verify-type fol
   ```

2. Run a small evaluation to ensure functionality:
   ```bash
   python run.py evaluate --data-dir reviseqa_data/nl/verified --batch-size 1
   ```

## Submitting Pull Requests

1. Update the README.md if needed
2. Add tests for new functionality
3. Ensure all tests pass
4. Write clear commit messages
5. Create a pull request with a clear description

## Reporting Issues

When reporting issues, please include:
- Python version
- Operating system
- Complete error messages
- Steps to reproduce the issue

## Code of Conduct

Please be respectful and professional in all interactions. We strive to maintain a welcoming environment for all contributors.

## Questions?

Feel free to open an issue for any questions about contributing!