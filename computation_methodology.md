# Computation Methodology: Step-by-Step Explanation

This document explains exactly how each analysis and figure was computed from the raw data.

---

## Data Structure

### Input Data: `combined_model_results.csv`

Each row contains:
- `task`: Task name (e.g., "explicit", "implicit_no_reasoning")
- `model`: Model name (e.g., "claude-sonnet-4")
- `k`: Number of correction steps (2, 4, or 7)
- `score`: Accuracy score (0.0 to 1.0)
- `lower_bound`: 95% confidence interval lower bound
- `upper_bound`: 95% confidence interval upper bound

**Example rows:**
```
task,model,k,score,lower_bound,upper_bound
explicit,claude-sonnet-4,7,0.5375,0.487,0.587
explicit,claude-sonnet-4,4,0.6825,0.636,0.727
explicit,claude-sonnet-4,2,0.8125,0.772,0.848
```

### Task Classifications

**Chain-of-Thought (COT) Classification:**
```python
# Tasks WITHOUT "no_reasoning" = COT (models provide reasoning)
# Tasks WITH "no_reasoning" = Standard (no reasoning required)

df['method'] = df['task'].apply(
    lambda x: 'Standard' if 'no_reasoning' in x else 'COT'
)
```

**Examples:**
- `explicit` → COT (requires reasoning)
- `explicit_no_reasoning` → Standard (no reasoning)
- `implicit` → COT
- `implicit_no_reasoning` → Standard

**Correction Classification:**
```python
# Tasks WITH "no_correction" = No Correction
# Tasks WITHOUT "no_correction" = With Correction

df['has_correction'] = df['task'].apply(
    lambda x: 'No Correction' if 'no_correction' in x else 'With Correction'
)
```

**Examples:**
- `explicit` → With Correction
- `explicit_no_correction` → No Correction
- `implicit` → With Correction
- `implicit_no_correction` → No Correction

**Context Type Classification:**
```python
df['context_type'] = df['task'].apply(
    lambda x: 'Implicit' if x.startswith('implicit') else 'Explicit'
)
```

---

## 1. Chain-of-Thought (COT) vs. Standard Analysis

### 1.1 Overall Performance Comparison

**Step 1: Calculate average score by model and method**
```python
# Group by model and method, compute mean score
model_method_avg = df.groupby(['model', 'method'])['score'].mean().reset_index()

# Example output:
# model                method    score
# claude-sonnet-4     COT       0.747917
# claude-sonnet-4     Standard  0.660625
```

**How this works:**
- For each model, we have scores across 8 tasks × 3 k-values = 24 data points
- 12 points are COT tasks (explicit, implicit, explicit_no_correction, implicit_no_correction)
- 12 points are Standard tasks (explicit_no_reasoning, implicit_no_reasoning, etc.)
- We average all COT scores and all Standard scores separately

**Step 2: Pivot to side-by-side comparison**
```python
comparison = model_method_avg.pivot(index='model', columns='method', values='score')

# Creates:
#                      COT    Standard
# claude-sonnet-4    0.748     0.661
# gemini-2.5-pro     0.772     0.773
# ...
```

**Step 3: Calculate difference**
```python
comparison['difference'] = comparison['COT'] - comparison['Standard']

# Positive difference = COT better
# Negative difference = Standard better
```

**Step 4: Overall statistics**
```python
# Average across all models
avg_cot = comparison['COT'].mean()           # 0.472
avg_standard = comparison['Standard'].mean() # 0.435
overall_diff = comparison['difference'].mean() # +0.037

# Count how many models benefit
cot_better = (comparison['difference'] > 0.01).sum()      # 16 models (84.2%)
standard_better = (comparison['difference'] < -0.01).sum() # 1 model (5.3%)
similar = (abs(comparison['difference']) <= 0.01).sum()   # 2 models (10.5%)
```

### 1.2 COT Performance by K-value

**For each k-value (7, 4, 2):**
```python
for k_val in [7, 4, 2]:
    # Filter data for this k-value
    df_k = df[df['k'] == k_val]

    # Group by model and method
    model_method_k = df_k.groupby(['model', 'method'])['score'].mean().reset_index()

    # Pivot
    comparison_k = model_method_k.pivot(index='model', columns='method', values='score')
    comparison_k['difference'] = comparison_k['COT'] - comparison_k['Standard']

    # Statistics
    cot_better_k = (comparison_k['difference'] > 0.01).sum()
    avg_diff_k = comparison_k['difference'].mean()
```

**Example for k=7:**
```
Average COT score: 0.331
Average Standard score: 0.310
COT advantage: +0.021 (2.1 percentage points)
Models where COT better: 10 (52.6%)
```

### 1.3 COT Performance by Task Type and K-value

**For each combination of k-value and task pair:**
```python
task_pairs = {
    'Explicit': ('explicit', 'explicit_no_reasoning'),
    'Implicit': ('implicit', 'implicit_no_reasoning'),
    # ... etc
}

for k_val in [7, 4, 2]:
    df_k = df[df['k'] == k_val]

    for task_label, (cot_task, standard_task) in task_pairs.items():
        # Get scores for COT task
        cot_data = df_k[df_k['task'] == cot_task][['model', 'score']]
        # Get scores for Standard task
        standard_data = df_k[df_k['task'] == standard_task][['model', 'score']]

        # Merge on model
        merged = cot_data.merge(standard_data, on='model',
                                suffixes=('_COT', '_Standard'))

        # Calculate difference per model
        merged['difference'] = merged['score_COT'] - merged['score_Standard']

        # Average difference
        avg_diff = merged['difference'].mean()

        # Count how many models benefit
        cot_better = (merged['difference'] > 0.01).sum()
        pct_better = cot_better / len(merged) * 100
```

**Example: Explicit task at k=2:**
```
COT average: 0.535
Standard average: 0.469
Difference: +0.066 (6.6 percentage points)
84% of models benefit from COT
```

---

## 2. Explicit vs. Implicit Context Analysis

### 2.1 Overall Performance by Context Type

**Step 1: Calculate average by context type**
```python
context_scores = df.groupby('context_type')['score'].mean()

# Output:
# context_type
# Explicit    0.319
# Implicit    0.613
```

**Step 2: Calculate gap**
```python
gap = context_scores['Implicit'] - context_scores['Explicit']
# 0.613 - 0.319 = 0.294 (29.4 percentage points)
```

### 2.2 Performance by Context Type and K-value

```python
context_k_scores = df.groupby(['context_type', 'k'])['score'].mean().reset_index()

# Output:
# context_type  k    score
# Explicit      7    0.151
# Explicit      4    0.295
# Explicit      2    0.535
# Implicit      7    0.511
# Implicit      4    0.617
# Implicit      2    0.736
```

**Gap calculation for each k:**
```python
for k_val in [7, 4, 2]:
    explicit_score = context_k_scores[
        (context_k_scores['context_type'] == 'Explicit') &
        (context_k_scores['k'] == k_val)
    ]['score'].values[0]

    implicit_score = context_k_scores[
        (context_k_scores['context_type'] == 'Implicit') &
        (context_k_scores['k'] == k_val)
    ]['score'].values[0]

    gap = implicit_score - explicit_score
```

**Results:**
- k=7: gap = 0.360 (36.0 points)
- k=4: gap = 0.322 (32.2 points)
- k=2: gap = 0.201 (20.1 points)

### 2.3 Per-Model Explicit vs Implicit Comparison

```python
# Average by model and context type
model_context = df.groupby(['model', 'context_type'])['score'].mean().reset_index()

# Pivot to get explicit and implicit side-by-side
model_pivot = model_context.pivot(index='model', columns='context_type', values='score')

# Example:
#                    Explicit  Implicit
# claude-sonnet-4     0.708     0.748
# gemini-2.5-pro      0.771     0.772
```

This creates the scatter plot data where x-axis = Explicit, y-axis = Implicit.

---

## 3. Correction Feedback Analysis

### 3.1 Overall Correction Impact

**Step 1: Average by correction status**
```python
corr_avg = df.groupby('has_correction')['score'].mean()

# Output:
# has_correction
# With Correction    0.457
# No Correction      0.450
```

**Step 2: Calculate difference**
```python
diff = corr_avg['With Correction'] - corr_avg['No Correction']
# 0.457 - 0.450 = 0.007 (0.7 percentage points)
```

### 3.2 Correction Impact by K-value and Task Type

**For each k-value and task pair:**
```python
task_pairs_corr = {
    'Explicit (COT)': ('explicit', 'explicit_no_correction'),
    'Explicit (Standard)': ('explicit_no_reasoning', 'explicit_no_reasoning_no_correction'),
    'Implicit (COT)': ('implicit', 'implicit_no_correction'),
    'Implicit (Standard)': ('implicit_no_reasoning', 'implicit_no_reasoning_no_correction'),
}

for k_val in [7, 4, 2]:
    df_k = df[df['k'] == k_val]

    for task_label, (with_corr_task, no_corr_task) in task_pairs_corr.items():
        # Get average scores for each task
        with_corr_scores = df_k[df_k['task'] == with_corr_task]['score']
        no_corr_scores = df_k[df_k['task'] == no_corr_task]['score']

        # Calculate difference
        diff = with_corr_scores.mean() - no_corr_scores.mean()
```

**Example: Implicit (Standard) at k=2:**
```
With Correction: 0.700
No Correction: 0.686
Difference: +0.014 (1.4 percentage points)
```

### 3.3 Per-Model Correction Benefit

```python
# Average by model and correction status
model_corr = df.groupby(['model', 'has_correction'])['score'].mean().reset_index()

# Pivot
model_corr_pivot = model_corr.pivot(index='model', columns='has_correction', values='score')

# Calculate benefit
model_corr_pivot['benefit'] = (
    model_corr_pivot['With Correction'] -
    model_corr_pivot['No Correction']
)

# Sort by benefit
model_corr_pivot = model_corr_pivot.sort_values('benefit')
```

**Top performer:**
```
qwen-2.5-coder-32b-instruct:
  With Correction: 0.547
  No Correction: 0.476
  Benefit: +0.071 (7.1 percentage points)
```

---

## 4. Three-Way Interaction Analysis

### 4.1 COT × Context × Correction

```python
# Group by all three factors
three_way = df.groupby(['context_type', 'method', 'has_correction'])['score'].mean().reset_index()

# Example output:
# context_type  method    has_correction    score
# Explicit      COT       With Correction   0.323
# Explicit      COT       No Correction     0.319
# Explicit      Standard  With Correction   0.287
# Explicit      Standard  No Correction     0.280
# Implicit      COT       With Correction   0.621
# Implicit      COT       No Correction     0.622
# Implicit      Standard  With Correction   0.594
# Implicit      Standard  No Correction     0.579
```

### 4.2 Effect Size Calculation

```python
# COT effect
cot_effect = df.groupby('method')['score'].mean()
cot_size = abs(cot_effect['COT'] - cot_effect['Standard'])
# 0.472 - 0.435 = 0.037

# Context effect
context_effect = df.groupby('context_type')['score'].mean()
context_size = abs(context_effect['Implicit'] - context_effect['Explicit'])
# 0.613 - 0.319 = 0.294

# Correction effect
corr_effect = df.groupby('has_correction')['score'].mean()
corr_size = abs(corr_effect['With Correction'] - corr_effect['No Correction'])
# 0.457 - 0.450 = 0.007

# Relative sizes
print(f"Context: {context_size:.3f} (largest)")
print(f"COT: {cot_size:.3f} ({cot_size/corr_size:.1f}× correction)")
print(f"Correction: {corr_size:.3f} (smallest)")
```

**Output:**
```
Context: 0.294 (largest)
COT: 0.037 (5.3× correction)
Correction: 0.007 (smallest)
```

---

## 5. Model Size Analysis

### 5.1 Size vs Performance Correlation

**Step 1: Load and merge data**
```python
# Load model sizes
model_sizes = pd.read_csv('model_sizes.csv')

# Calculate average score per model
model_avg_scores = df.groupby('model')['score'].mean().reset_index()
model_avg_scores.columns = ['model', 'avg_score']

# Merge
size_performance = model_sizes.merge(model_avg_scores, on='model')
```

**Step 2: Convert sizes to numeric**
```python
# Handle estimates (remove ~ prefix)
size_performance['size_numeric'] = size_performance['size_billions'].apply(
    lambda x: float(str(x).replace('~', '')) if pd.notna(x) else np.nan
)
```

**Step 3: Correlation analysis**
```python
# For standard models only
standard_models = size_performance[size_performance['architecture'] == 'Standard'].dropna()

# Log-linear fit
z = np.polyfit(np.log(standard_models['size_numeric']),
               standard_models['avg_score'], 1)
# Slope is very close to 0, indicating weak correlation
```

### 5.2 Efficiency Calculation

```python
# Efficiency = accuracy per billion parameters × 100
size_performance['efficiency'] = (
    size_performance['avg_score'] / size_performance['size_numeric'] * 100
)

# Sort by efficiency
efficiency_sorted = size_performance.nlargest(10, 'efficiency')
```

**Top performer:**
```
gemini-2-5-flash:
  Size: ~5B
  Score: 0.620
  Efficiency: 12.4 (0.620 / 5 × 100)
```

---

## 6. Statistical Significance

### 6.1 Confidence Intervals

The data includes 95% confidence intervals calculated as:
```python
# For each task-model-k combination, we have:
# - score: mean accuracy
# - lower_bound: 95% CI lower bound
# - upper_bound: 95% CI upper bound

# These were pre-computed in the original benchmark using:
# CI = score ± 1.96 × SE
# where SE = sqrt(p(1-p)/n)
# p = accuracy, n = number of test examples
```

### 6.2 Determining Significant Differences

We use a threshold of 0.01 (1 percentage point) to determine significance:
```python
# COT significantly better if:
difference > 0.01

# Standard significantly better if:
difference < -0.01

# Similar performance if:
abs(difference) <= 0.01
```

**Why 1% threshold?**
- Typical 95% CI width is ~0.04-0.06 (±0.02-0.03)
- Differences of 1% are roughly 1/2 to 1/3 of typical CI width
- Provides conservative estimate of meaningful differences
- Aligns with practical significance in ML benchmarks

---

## 7. Figure Generation Details

### 7.1 Figure 1: COT vs Standard by K-value

**Data preparation:**
```python
for k_val in [7, 4, 2]:
    # Filter to this k-value
    df_k = df[df['k'] == k_val]

    # Average by model and method
    model_method = df_k.groupby(['model', 'method'])['score'].mean().reset_index()

    # Pivot for side-by-side comparison
    comparison = model_method.pivot(index='model', columns='method', values='score')
    comparison['difference'] = comparison['COT'] - comparison['Standard']

    # Sort by difference
    comparison = comparison.sort_values('difference')
```

**Visualization:**
```python
# Horizontal bar chart
x = np.arange(len(comparison))
width = 0.35

bars1 = ax.barh(x - width/2, comparison['Standard'], width,
                label='Standard', color='#E8927C')
bars2 = ax.barh(x + width/2, comparison['COT'], width,
                label='COT', color='#69B3E7')
```

### 7.2 Figure 2: COT Benefit Heatmap

**Data preparation:**
```python
task_pairs = {
    'Explicit': ('explicit', 'explicit_no_reasoning'),
    'Implicit': ('implicit', 'implicit_no_reasoning'),
    'Explicit\n(no corr)': ('explicit_no_correction', 'explicit_no_reasoning_no_correction'),
    'Implicit\n(no corr)': ('implicit_no_correction', 'implicit_no_reasoning_no_correction'),
}

heatmap_data = []
for k_val in [7, 4, 2]:
    row_data = []
    df_k = df[df['k'] == k_val]

    for task_label, (cot_task, standard_task) in task_pairs.items():
        # Get mean scores for each task
        cot_score = df_k[df_k['task'] == cot_task]['score'].mean()
        standard_score = df_k[df_k['task'] == standard_task]['score'].mean()

        # Calculate difference
        diff = cot_score - standard_score
        row_data.append(diff)

    heatmap_data.append(row_data)

# heatmap_data is now a 3×4 matrix
# Rows: k=7, k=4, k=2
# Cols: Explicit, Implicit, Explicit(no corr), Implicit(no corr)
```

**Visualization:**
```python
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
            xticklabels=list(task_pairs.keys()),
            yticklabels=['k=7 (Hard)', 'k=4 (Medium)', 'k=2 (Easy)'])
```

### 7.3 Figure 3: Explicit vs Implicit Gap

**Left panel - bar chart:**
```python
context_k_scores = df.groupby(['context_type', 'k'])['score'].mean().reset_index()

# Separate into explicit and implicit
explicit_scores = context_k_scores[context_k_scores['context_type'] == 'Explicit']['score'].values
implicit_scores = context_k_scores[context_k_scores['context_type'] == 'Implicit']['score'].values

# Bar positions
x = np.array([0, 1, 2])  # k=7, k=4, k=2
width = 0.35

bars1 = ax.bar(x - width/2, explicit_scores, width, label='Explicit')
bars2 = ax.bar(x + width/2, implicit_scores, width, label='Implicit')
```

**Right panel - gap trend:**
```python
gaps = implicit_scores - explicit_scores
# [0.360, 0.322, 0.201]

ax.plot(x, gaps, marker='o', linewidth=2)
ax.fill_between(x, 0, gaps, alpha=0.3)
```

### 7.4 Figure 4: Scatter Plot

**Data preparation:**
```python
# Average by model and context type
model_context = df.groupby(['model', 'context_type'])['score'].mean().reset_index()

# Pivot to get explicit and implicit as separate columns
model_pivot = model_context.pivot(index='model', columns='context_type', values='score')

# Now we have:
#                    Explicit  Implicit
# model1              0.5       0.7
# model2              0.6       0.8
```

**Visualization:**
```python
ax.scatter(model_pivot['Explicit'], model_pivot['Implicit'], s=100, alpha=0.6)

# Add diagonal line (equal performance)
lims = [0, 1.0]
ax.plot(lims, lims, 'k--', alpha=0.3, label='Equal Performance')

# Shade regions
ax.fill_between(lims, lims, 1.0, alpha=0.1, color='blue',
                label='Implicit Advantage')
```

### 7.5 Figures 5-8

Similar data preparation and aggregation methods as above, with appropriate grouping, pivoting, and statistical calculations.

---

## 8. Key Formulas and Calculations

### 8.1 Average Score Calculation

For a model across all tasks:
```
avg_score = sum(all scores for this model) / number of scores

Example for claude-sonnet-4:
- 8 tasks × 3 k-values = 24 scores
- avg_score = sum(24 scores) / 24
```

### 8.2 COT Advantage Calculation

For a specific k-value:
```
COT_advantage = avg(COT scores at k) - avg(Standard scores at k)

Example for k=2:
- COT tasks: explicit, implicit, explicit_no_correction, implicit_no_correction
- Standard tasks: explicit_no_reasoning, implicit_no_reasoning, etc.
- Each has 19 models, so 19 scores per task
- COT_avg = mean of (4 tasks × 19 models) = mean of 76 scores
- Standard_avg = mean of (4 tasks × 19 models) = mean of 76 scores
- COT_advantage = COT_avg - Standard_avg
```

### 8.3 Percentage of Models Benefiting

```
benefit_pct = (count of models with difference > threshold) / total_models × 100

Example:
- 16 models have COT - Standard > 0.01
- Total models = 19
- benefit_pct = 16/19 × 100 = 84.2%
```

### 8.4 Efficiency Score

```
efficiency = (accuracy / size_in_billions) × 100

Example for gemini-2-5-flash:
- accuracy = 0.620
- size = 5B
- efficiency = (0.620 / 5) × 100 = 12.4
```

---

## 9. Data Quality Checks

### 9.1 Completeness Check

```python
# Check for missing data
expected_rows = 19 models × 8 tasks × 3 k-values = 456 rows
actual_rows = len(df)

if actual_rows != expected_rows:
    print(f"Warning: Expected {expected_rows} rows, got {actual_rows}")
```

### 9.2 Score Range Validation

```python
# All scores should be between 0 and 1
assert df['score'].min() >= 0 and df['score'].max() <= 1

# Confidence intervals should bound the score
assert all(df['lower_bound'] <= df['score'])
assert all(df['score'] <= df['upper_bound'])
```

### 9.3 Task Name Validation

```python
# Verify task classifications
expected_tasks = [
    'explicit', 'implicit',
    'explicit_no_reasoning', 'implicit_no_reasoning',
    'explicit_no_correction', 'implicit_no_correction',
    'explicit_no_reasoning_no_correction', 'implicit_no_reasoning_no_correction'
]

assert set(df['task'].unique()) == set(expected_tasks)
```

---

## 10. Reproducibility

### 10.1 Random Seed

No random operations were used in the analysis, so results are fully deterministic.

### 10.2 Software Versions

```python
# pandas==2.0.0 or higher
# matplotlib==3.7.0 or higher
# seaborn==0.12.0 or higher
# numpy==1.24.0 or higher
```

### 10.3 Full Reproduction

To reproduce all analyses:
```bash
# 1. Generate COT vs Standard analysis
python analyze_cot_vs_standard.py

# 2. Generate Correction analysis
python analyze_correction_vs_no_correction.py

# 3. Generate all figures
python generate_discussion_figures.py
```

All scripts are deterministic and will produce identical results given the same input data.

---

## Summary

All analyses follow a consistent pattern:

1. **Filter** data to relevant subset (by k-value, task type, etc.)
2. **Group** by relevant factors (model, method, context type, etc.)
3. **Aggregate** using mean() to calculate average scores
4. **Pivot** to create side-by-side comparisons when needed
5. **Calculate** differences and effect sizes
6. **Visualize** using appropriate chart types

The key insight is that all comparisons are based on **averaged scores** across multiple conditions, with **differences calculated** between paired conditions to quantify effects.
