# ReviseQA Dataset Analysis - ACL Paper Summary

## Dataset Overview

- **Total Examples**: 930 verified examples (non-truncated)
- **Total Edits**: 6,510 contextual modifications
- **Total Verification Tasks**: 19,530 (6,510 edits × 3 models)

## Dataset Characteristics

### Original Examples
- **True conclusions**: 453 (48.7%)
- **False conclusions**: 473 (50.9%)
- **Uncertain**: 4 (0.4%)
- **Balance**: Near-perfect 50/50 split between True/False

### Context Complexity
| Metric | Mean ± SD | Min/Max |
|--------|-----------|---------|
| Context Statements | 13.00 ± 2.18 | 8 / 19 |
| Unique Predicates | 13.36 ± 2.23 | 8 / 19 |
| Reasoning Steps | 7.50 ± 1.14 | 6 / 10 |

### Reasoning Chain Analysis
- **Total reasoning steps across dataset**: 6,972
- **Steps with conclusions**: 6,628 (95.1%)
- **Steps without conclusions**: 344 (4.9%)
- **Average facts per step**: 1.57 ± 0.52
- **Average rules per step**: 0.98 ± 0.15

## Edit Characteristics

### Modification Types
- **FLIP edits**: 3,356 (51.6%)
  - Changes that flip the truth value of the conclusion
- **INVARIANT edits**: 3,153 (48.4%)
  - Changes that preserve the truth value despite context modification
- **UNCERTAIN**: 1 (0.0%)

### Answer Transition Analysis
- **Answer flipped**: 3,256 edits (50.0%)
- **Answer preserved**: 3,254 edits (50.0%)

**Most common transitions:**
1. False → True: 1,641
2. True → False: 1,587
3. Uncertain → False: 16
4. Uncertain → True: 12

### Edit Operations (Average per Edit)
- **Facts Removed**: 0.35
- **Rules Removed**: 0.63
- **Facts Added**: 1.03
- **Rules Added**: 1.11

### Top Edit Patterns
| Pattern | FLIP | INVARIANT | Total |
|---------|------|-----------|-------|
| R_F:0 R_R:1 A_F:0 A_R:1 | 850 | 67 | 917 |
| R_F:0 R_R:0 A_F:0 A_R:1 | 28 | 307 | 335 |
| R_F:1 R_R:0 A_F:1 A_R:0 | 321 | 5 | 326 |
| R_F:1 R_R:0 A_F:0 A_R:1 | 265 | 1 | 266 |
| R_F:0 R_R:1 A_F:1 A_R:1 | 98 | 139 | 237 |

*Note: R_F = Removed Facts, R_R = Removed Rules, A_F = Added Facts, A_R = Added Rules*

## Logical Complexity

### Operator Usage in FOL
| Operator | Frequency |
|----------|-----------|
| → (implies) | 5,120 |
| ⊕ (xor) | 2,963 |
| ¬ (not) | 2,819 |
| ∀ (forall) | 2,096 |
| ∨ (or) | 1,879 |
| ∧ (and) | 1,298 |
| ∃ (exists) | 0 |

## Model Performance

### Verification Accuracy
| Model | Verified/Total | Accuracy (%) |
|-------|----------------|--------------|
| google/gemini-2.5-flash | 6,465/6,510 | 99.3% |
| openai/gpt-5-mini | 6,438/6,510 | 98.9% |
| qwen/qwen3-235b-a22b-2507 | 5,381/6,510 | 82.7% |

### Error Analysis
- **gemini-2.5-flash**: 45 failures (0.7% error rate)
- **gpt-5-mini**: 72 failures (1.1% error rate)
- **qwen3-235b-a22b-2507**: 1,129 failures (17.3% error rate)

**Common error patterns:**
- Misinterpretation of inclusive vs. exclusive OR
- Incorrect handling of XOR (⊕) representations
- Universal quantification scope issues

## Key Statistics for Paper

### Abstract/Introduction
- Dataset of **930 logical reasoning examples** with **6,510 contextual modifications**
- Average of **13 context statements** and **7.5 reasoning steps** per example
- **Balanced dataset**: 48.7% True, 50.9% False conclusions
- **19,530 verification tasks** evaluated across 3 state-of-the-art models

### Results Section
- **Best performing models**: Gemini-2.5-Flash (99.3%) and GPT-5-Mini (98.9%)
- **Challenge for smaller models**: Qwen3-235B-A22B-2507 achieves 82.7%
- **Edit diversity**: 51.6% FLIP vs 48.4% INVARIANT modifications
- **Answer stability**: 50% of edits preserve original answer, 50% flip it

### Dataset Complexity
- Examples feature complex logical structures with average **13.36 unique predicates**
- Rich operator usage: **5,120 implications**, **2,963 XORs**, **2,096 universal quantifiers**
- Multi-step reasoning chains with **95.1% productive steps** (yielding conclusions)

## Files Generated

### Visualizations (PDF + PNG)
1. `dataset_distributions.pdf` - Distribution of context length, predicates, and reasoning steps
2. `modification_types.pdf` - Pie chart of FLIP vs INVARIANT modifications
3. `model_verification.pdf` - Bar chart comparing model verification rates
4. `edit_correlations.pdf` - Heatmap of edit operation correlations

### Tables (LaTeX)
1. `dataset_table.tex` - Main dataset statistics table
2. `model_performance_table.tex` - Model verification performance table

### Data Exports (CSV)
1. `dataset_statistics.csv` - Summary statistics for all examples
2. `detailed_dataset_export.csv` - Complete edit-level data export

### Reports
1. `paper_summary.txt` - Quick reference summary
2. `PAPER_READY_SUMMARY.md` - This comprehensive report

## Suggested Paper Sections

### Dataset Description
> "ReviseQA consists of 930 verified logical reasoning examples with 6,510 contextual modifications. Each example contains an average of 13 context statements expressed in both natural language and first-order logic, with an average of 13.36 unique predicates. The reasoning chains comprise an average of 7.5 steps, demonstrating multi-hop logical inference capabilities. The dataset is balanced with 48.7% True and 50.9% False conclusions."

### Edit Strategy
> "We generate two types of modifications: FLIP edits (51.6%) that change the conclusion's truth value, and INVARIANT edits (48.4%) that preserve it despite context changes. The most common edit pattern involves removing one rule and adding one rule (917 instances), demonstrating our focus on semantic rather than syntactic modifications."

### Evaluation Complexity
> "The logical complexity of our dataset is evidenced by rich operator usage: 5,120 implications, 2,963 exclusive disjunctions (XOR), and 2,096 universal quantifiers. This diversity challenges models to handle various logical constructs including conditional reasoning, mutual exclusion, and quantification."

### Model Results
> "State-of-the-art models achieve high verification accuracy, with Gemini-2.5-Flash (99.3%) and GPT-5-Mini (98.9%) demonstrating strong capabilities. However, the smaller Qwen3-235B-A22B-2507 model achieves 82.7%, highlighting remaining challenges. Common errors include misinterpretation of inclusive vs. exclusive OR and universal quantification scope."

---

**Generated**: October 4, 2025
**Analysis Scripts**: `dataset_analysis.py`, `dataset_analysis_paper.py`, `dataset_analysis_detailed.py`
