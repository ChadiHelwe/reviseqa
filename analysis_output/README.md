# ReviseQA Dataset Analysis - Output Guide

This directory contains comprehensive analysis of the ReviseQA dataset for your ACL paper.

## 📊 Quick Start - Paper Ready Materials

### Primary Summary Document
- **`PAPER_READY_SUMMARY.md`** - Complete analysis with all statistics, suggested paper sections, and key findings

### LaTeX Tables (Ready to Insert)
- **`dataset_table.tex`** - Main dataset statistics table
- **`model_performance_table.tex`** - Model verification performance table

### Publication-Quality Figures
1. **`dataset_overview_comprehensive.pdf`** - Complete dataset composition (6-panel figure)
2. **`dataset_distributions.pdf`** - Context length, predicates, and reasoning steps
3. **`modification_types.pdf`** - FLIP vs INVARIANT breakdown
4. **`model_verification.pdf`** - Model performance comparison
5. **`edit_correlations.pdf`** - Edit operation correlation heatmap

*All figures also available as high-resolution PNG files*

## 📁 File Organization

### Summary Documents
| File | Description |
|------|-------------|
| `PAPER_READY_SUMMARY.md` | Comprehensive analysis with paper-ready text |
| `paper_summary.txt` | Quick reference statistics |

### Data Exports
| File | Description |
|------|-------------|
| `dataset_statistics.csv` | Summary statistics for all examples |
| `detailed_dataset_export.csv` | Complete edit-level export (628 KB) |

### Visualizations (PDF + PNG)
| File | Content |
|------|---------|
| `dataset_overview_comprehensive.*` | 6-panel overview figure |
| `dataset_distributions.*` | Distribution of key metrics |
| `modification_types.*` | Pie chart of edit types |
| `model_verification.*` | Model performance bar chart |
| `edit_correlations.*` | Correlation heatmap |

### LaTeX Tables
| File | Content |
|------|---------|
| `dataset_table.tex` | Dataset statistics table |
| `model_performance_table.tex` | Model accuracy table |

## 🎯 Key Statistics at a Glance

### Dataset Scale
- **930** examples
- **6,510** edits
- **19,530** verification tasks

### Balance
- **48.7%** True conclusions
- **50.9%** False conclusions
- **51.6%** FLIP edits
- **48.4%** INVARIANT edits

### Complexity
- **13.00 ± 2.18** context statements per example
- **13.36 ± 2.23** unique predicates per example
- **7.50 ± 1.14** reasoning steps per example

### Model Performance
- **Gemini-2.5-Flash**: 99.3% (6,465/6,510)
- **GPT-5-Mini**: 98.9% (6,438/6,510)
- **Qwen3-235B**: 82.7% (5,381/6,510)

## 📝 Using These Materials in Your Paper

### In Abstract/Introduction
```
We present ReviseQA, a dataset of 930 logical reasoning examples with 6,510
contextual modifications. Each example contains an average of 13 context
statements and 7.5 reasoning steps, featuring complex logical operators
including 5,120 implications and 2,963 exclusive disjunctions.
```

### In Dataset Section
Use `dataset_table.tex` directly in your LaTeX document:
```latex
\input{tables/dataset_table.tex}
```

### In Results Section
Use `model_performance_table.tex` for model comparison:
```latex
\input{tables/model_performance_table.tex}
```

### For Figures
All figures are publication-ready at 300 DPI:
```latex
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/dataset_overview_comprehensive.pdf}
\caption{ReviseQA dataset composition and statistics.}
\label{fig:dataset_overview}
\end{figure}
```

## 🔬 Detailed Analysis Highlights

### Edit Patterns
- Most common: **Remove 1 rule, Add 1 rule** (917 instances)
- Second: **Add 1 rule only** (335 instances)
- Third: **Remove 1 fact, Add 1 fact** (326 instances)

### Logical Operators
- **5,120** implications (→)
- **2,963** exclusive OR (⊕)
- **2,819** negations (¬)
- **2,096** universal quantifiers (∀)
- **0** existential quantifiers (∃)

### Reasoning Chains
- **6,972** total reasoning steps
- **95.1%** steps yield conclusions
- **1.57 ± 0.52** facts per step
- **0.98 ± 0.15** rules per step

### Answer Flips
- **50.0%** of edits flip the answer
- **50.0%** preserve the answer
- **False → True**: 1,641 transitions
- **True → False**: 1,587 transitions

## 🛠️ Regenerating Analysis

If you need to regenerate or modify the analysis:

```bash
# Basic statistics
python dataset_analysis.py

# Paper-ready figures and tables
python dataset_analysis_paper.py

# Detailed breakdowns
python dataset_analysis_detailed.py

# Overview figure
python create_dataset_overview_figure.py
```

## 📊 Data Access

### CSV Exports
1. **`dataset_statistics.csv`** - Example-level statistics (930 rows)
   - Columns: num_context_statements, num_predicates, reasoning_steps, original_answer, num_edits

2. **`detailed_dataset_export.csv`** - Edit-level statistics (6,510 rows)
   - Columns: example_id, edit_id, modification_type, removed/added facts/rules, model verification results

## 📚 Citation Suggestion

```bibtex
@inproceedings{yourname2025reviseqa,
  title={ReviseQA: A Dataset for Evaluating Logical Reasoning through Context Revision},
  author={Your Name and Collaborators},
  booktitle={Proceedings of ACL 2025},
  year={2025},
  note={Dataset: 930 examples, 6,510 edits, 19,530 verification tasks}
}
```

## 📞 Questions?

All analysis scripts are documented and can be modified as needed. The dataset is located in:
```
reviseqa_data/nl/verified/
```

Only non-truncated files (930 total) were analyzed.

---

**Generated**: October 4, 2025
**Analysis Version**: 1.0
**Scripts**: `dataset_analysis.py`, `dataset_analysis_paper.py`, `dataset_analysis_detailed.py`
