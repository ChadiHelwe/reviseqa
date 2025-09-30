#!/usr/bin/env python3
"""
Comprehensive error analysis of models in detailed_models_results directory.
"""

import json
import os
import re
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import statistics

@dataclass
class PredictionResult:
    step: int
    prediction: str
    correct_answer: str
    reasoning: str
    correct: bool
    tags: List[str]
    token_count: Optional[int] = None

@dataclass
class ModelTaskResult:
    model_name: str
    task_path: str
    chain_index: int
    include_reasoning: bool
    include_correction: bool
    total_steps: int
    final_accuracy: float
    predictions: List[PredictionResult]

class ModelAnalyzer:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.results: List[ModelTaskResult] = []
        self.model_stats = defaultdict(dict)

    def load_all_results(self):
        """Load all JSON files from the directory structure."""
        print("Loading all model results...")
        count = 0

        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)

                        # Parse predictions
                        predictions = []
                        for pred in data.get('predictions', []):
                            predictions.append(PredictionResult(
                                step=pred.get('step', 0),
                                prediction=pred.get('prediction', ''),
                                correct_answer=pred.get('correct_answer', ''),
                                reasoning=pred.get('reasoning', ''),
                                correct=pred.get('correct', False),
                                tags=pred.get('tags', []),
                                token_count=pred.get('token_count')
                            ))

                        # Create result object
                        metadata = data.get('metadata', {})
                        result = ModelTaskResult(
                            model_name=metadata.get('model_name', ''),
                            task_path=metadata.get('task_path', ''),
                            chain_index=metadata.get('chain_index', 0),
                            include_reasoning=metadata.get('include_reasoning', False),
                            include_correction=metadata.get('include_correction', False),
                            total_steps=metadata.get('total_steps', 0),
                            final_accuracy=metadata.get('final_accuracy', 0.0),
                            predictions=predictions
                        )

                        self.results.append(result)
                        count += 1

                        if count % 100 == 0:
                            print(f"Loaded {count} files...")

                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")

        print(f"Total files loaded: {count}")

    def analyze_prediction_patterns(self):
        """Analyze prediction patterns for each model and task combination."""
        print("\nAnalyzing prediction patterns...")

        model_task_stats = defaultdict(lambda: defaultdict(lambda: {
            'total_predictions': 0,
            'true_predictions': 0,
            'false_predictions': 0,
            'uncertain_predictions': 0,
            'correct_predictions': 0,
            'accuracy': 0.0,
            'chains': set()
        }))

        for result in self.results:
            key = f"{result.model_name}_{result.task_path}"
            stats = model_task_stats[result.model_name][result.task_path]
            stats['chains'].add(result.chain_index)

            for pred in result.predictions:
                stats['total_predictions'] += 1

                # Handle both string and boolean prediction values
                if isinstance(pred.prediction, bool):
                    pred_str = str(pred.prediction).lower()
                else:
                    pred_str = str(pred.prediction).lower()

                if pred_str == 'true':
                    stats['true_predictions'] += 1
                elif pred_str == 'false':
                    stats['false_predictions'] += 1
                elif pred_str == 'uncertain':
                    stats['uncertain_predictions'] += 1

                if pred.correct:
                    stats['correct_predictions'] += 1

        # Calculate accuracies
        for model in model_task_stats:
            for task in model_task_stats[model]:
                stats = model_task_stats[model][task]
                if stats['total_predictions'] > 0:
                    stats['accuracy'] = stats['correct_predictions'] / stats['total_predictions']
                stats['chains'] = len(stats['chains'])

        self.model_task_stats = model_task_stats
        return model_task_stats

    def analyze_error_types(self):
        """Categorize errors by type."""
        print("Analyzing error types...")

        error_categories = {
            'logical_reasoning': [],
            'uncertainty_handling': [],
            'context_comprehension': [],
            'consistency_errors': []
        }

        for result in self.results:
            for pred in result.predictions:
                if not pred.correct:
                    # Handle different data types
                    pred_str = str(pred.prediction).lower()
                    correct_str = str(pred.correct_answer).lower()

                    error_info = {
                        'model': result.model_name,
                        'task': result.task_path,
                        'prediction': pred_str,
                        'correct_answer': correct_str,
                        'reasoning': pred.reasoning,
                        'step': pred.step
                    }

                    # Categorize based on reasoning patterns
                    reasoning_lower = pred.reasoning.lower()

                    # Uncertainty handling errors
                    if (pred_str == 'uncertain' and correct_str in ['true', 'false']) or \
                       (pred_str in ['true', 'false'] and correct_str == 'uncertain'):
                        error_categories['uncertainty_handling'].append(error_info)

                    # Look for logical reasoning issues
                    elif any(keyword in reasoning_lower for keyword in ['therefore', 'implies', 'contradiction', 'inconsistent']):
                        error_categories['logical_reasoning'].append(error_info)

                    # Context comprehension
                    elif any(keyword in reasoning_lower for keyword in ['context', 'given', 'states', 'rule']):
                        error_categories['context_comprehension'].append(error_info)

                    # Default to consistency errors
                    else:
                        error_categories['consistency_errors'].append(error_info)

        self.error_categories = error_categories
        return error_categories

    def analyze_reasoning_quality(self):
        """Analyze reasoning quality patterns."""
        print("Analyzing reasoning quality...")

        reasoning_analysis = defaultdict(lambda: {
            'clear_reasoning_count': 0,
            'unclear_reasoning_count': 0,
            'avg_reasoning_length': 0,
            'common_mistakes': Counter(),
            'step_by_step_count': 0
        })

        for result in self.results:
            model_analysis = reasoning_analysis[result.model_name]

            for pred in result.predictions:
                reasoning = pred.reasoning

                # Analyze reasoning clarity
                if len(reasoning) > 50 and any(word in reasoning.lower() for word in ['therefore', 'because', 'since', 'given']):
                    model_analysis['clear_reasoning_count'] += 1
                else:
                    model_analysis['unclear_reasoning_count'] += 1

                # Check for step-by-step reasoning
                if '..' in reasoning or reasoning.count('.') > 3:
                    model_analysis['step_by_step_count'] += 1

                # Track common mistake patterns
                if not pred.correct:
                    if 'contradiction' in reasoning.lower():
                        model_analysis['common_mistakes']['contradiction_handling'] += 1
                    if 'uncertain' in str(pred.prediction).lower():
                        model_analysis['common_mistakes']['excessive_uncertainty'] += 1
                    if len(reasoning) < 30:
                        model_analysis['common_mistakes']['insufficient_reasoning'] += 1

        # Calculate averages
        for model in reasoning_analysis:
            total_predictions = reasoning_analysis[model]['clear_reasoning_count'] + reasoning_analysis[model]['unclear_reasoning_count']
            if total_predictions > 0:
                reasoning_analysis[model]['clarity_ratio'] = reasoning_analysis[model]['clear_reasoning_count'] / total_predictions

        self.reasoning_analysis = reasoning_analysis
        return reasoning_analysis

    def generate_performance_comparison(self):
        """Generate performance comparison across models."""
        print("Generating performance comparison...")

        # Overall accuracy by model
        model_accuracies = {}
        task_performance = defaultdict(dict)
        uncertain_rates = defaultdict(float)

        for model in self.model_task_stats:
            total_correct = 0
            total_predictions = 0
            total_uncertain = 0

            for task in self.model_task_stats[model]:
                stats = self.model_task_stats[model][task]
                total_correct += stats['correct_predictions']
                total_predictions += stats['total_predictions']
                total_uncertain += stats['uncertain_predictions']

                task_performance[task][model] = stats['accuracy']

            if total_predictions > 0:
                model_accuracies[model] = total_correct / total_predictions
                uncertain_rates[model] = total_uncertain / total_predictions

        # Find best/worst performers
        best_model = max(model_accuracies.items(), key=lambda x: x[1]) if model_accuracies else ("None", 0)
        worst_model = min(model_accuracies.items(), key=lambda x: x[1]) if model_accuracies else ("None", 0)

        self.performance_summary = {
            'model_accuracies': model_accuracies,
            'task_performance': dict(task_performance),
            'uncertain_rates': dict(uncertain_rates),
            'best_model': best_model,
            'worst_model': worst_model
        }

        return self.performance_summary

    def generate_report(self):
        """Generate comprehensive analysis report."""
        print("\nGenerating comprehensive report...")

        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE MODEL ERROR ANALYSIS REPORT")
        report.append("=" * 80)

        # Models and tasks summary
        report.append(f"\n1. MODELS AND TASKS IDENTIFIED")
        report.append("-" * 40)

        all_models = set()
        all_tasks = set()
        for result in self.results:
            all_models.add(result.model_name)
            all_tasks.add(result.task_path)

        report.append(f"Total Models: {len(all_models)}")
        for model in sorted(all_models):
            report.append(f"  - {model}")

        report.append(f"\nTotal Task Types: {len(all_tasks)}")
        for task in sorted(all_tasks):
            report.append(f"  - {task}")

        # Prediction patterns
        report.append(f"\n2. PREDICTION PATTERNS ANALYSIS")
        report.append("-" * 40)

        for model in sorted(self.model_task_stats.keys()):
            report.append(f"\nModel: {model}")
            for task in sorted(self.model_task_stats[model].keys()):
                stats = self.model_task_stats[model][task]
                report.append(f"  Task: {task}")
                report.append(f"    Total Predictions: {stats['total_predictions']}")
                report.append(f"    TRUE: {stats['true_predictions']} | FALSE: {stats['false_predictions']} | UNCERTAIN: {stats['uncertain_predictions']}")
                report.append(f"    Accuracy: {stats['accuracy']:.3f}")
                report.append(f"    Chains: {stats['chains']}")

        # Error analysis
        report.append(f"\n3. ERROR TYPE ANALYSIS")
        report.append("-" * 40)

        for error_type, errors in self.error_categories.items():
            report.append(f"\n{error_type.replace('_', ' ').title()}: {len(errors)} errors")

            if errors:
                # Show example
                example = errors[0]
                report.append(f"  Example - Model: {example['model']}")
                report.append(f"  Predicted: {example['prediction']} | Correct: {example['correct_answer']}")
                report.append(f"  Reasoning: {example['reasoning'][:200]}...")

        # Reasoning quality
        report.append(f"\n4. REASONING QUALITY ASSESSMENT")
        report.append("-" * 40)

        for model in sorted(self.reasoning_analysis.keys()):
            analysis = self.reasoning_analysis[model]
            report.append(f"\nModel: {model}")
            report.append(f"  Clear Reasoning: {analysis['clear_reasoning_count']}")
            report.append(f"  Unclear Reasoning: {analysis['unclear_reasoning_count']}")
            report.append(f"  Clarity Ratio: {analysis.get('clarity_ratio', 0):.3f}")
            report.append(f"  Step-by-step Count: {analysis['step_by_step_count']}")

            if analysis['common_mistakes']:
                report.append("  Common Mistakes:")
                for mistake, count in analysis['common_mistakes'].most_common(3):
                    report.append(f"    {mistake}: {count}")

        # Performance comparison
        report.append(f"\n5. PERFORMANCE COMPARISON")
        report.append("-" * 40)

        perf = self.performance_summary
        report.append(f"\nOverall Model Accuracies:")
        for model, accuracy in sorted(perf['model_accuracies'].items(), key=lambda x: x[1], reverse=True):
            report.append(f"  {model}: {accuracy:.3f}")

        report.append(f"\nBest Performing Model: {perf['best_model'][0]} ({perf['best_model'][1]:.3f})")
        report.append(f"Worst Performing Model: {perf['worst_model'][0]} ({perf['worst_model'][1]:.3f})")

        report.append(f"\nUncertain Prediction Rates:")
        for model, rate in sorted(perf['uncertain_rates'].items(), key=lambda x: x[1], reverse=True):
            report.append(f"  {model}: {rate:.3f}")

        report.append(f"\nTask-Specific Performance:")
        for task in sorted(perf['task_performance'].keys()):
            report.append(f"\n  {task}:")
            task_perfs = perf['task_performance'][task]
            for model, accuracy in sorted(task_perfs.items(), key=lambda x: x[1], reverse=True):
                report.append(f"    {model}: {accuracy:.3f}")

        return "\n".join(report)

def main():
    base_path = "/Users/helwec/Desktop/reviseqa/detailed_models_results"

    analyzer = ModelAnalyzer(base_path)
    analyzer.load_all_results()
    analyzer.analyze_prediction_patterns()
    analyzer.analyze_error_types()
    analyzer.analyze_reasoning_quality()
    analyzer.generate_performance_comparison()

    report = analyzer.generate_report()

    # Save report
    with open('/Users/helwec/Desktop/reviseqa/model_analysis_report.txt', 'w') as f:
        f.write(report)

    print("\nReport saved to model_analysis_report.txt")
    print("\nSample of report:")
    print(report[:2000] + "...\n[truncated]")

if __name__ == "__main__":
    main()