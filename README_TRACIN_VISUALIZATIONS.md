# TracIn Visualization Tools

This directory contains tools for visualizing TracIn influence scores, which help analyze the impact of training samples on model performance.

## Overview

These visualization tools analyze the influence scores computed by the TracIn method from the paper ["Estimating Training Data Influence by Tracing Gradient Descent"](https://arxiv.org/abs/2002.08484). They provide insights into:

1. **Self-influence** - How difficult each training sample is for the model to learn
2. **Test influence** - How training samples affect the model's predictions on test samples

## Available Scripts

### 1. `visualize_influence.py`

Generates visualizations for self-influence scores, showing which training samples are most difficult for the model.

**Usage:**
```bash
python visualize_influence.py
```

**Output:**
- `self_influence_by_pair.png` - Bar chart of average self-influence by angle pair
- `self_influence_heatmap.png` - Heatmap showing influence by degree combination
- `self_influence_distribution.png` - Histogram of self-influence scores
- `top_bottom_influential_pairs.png` - Most and least influential training pairs
- `influence_by_angle_diff.png` - Self-influence vs angle difference

### 2. `visualize_test_influence.py`

Analyzes how training samples influence model predictions on test samples.

**Usage:**
```bash
python visualize_test_influence.py
```

**Output:**
- `test_influence_distributions.png` - Distributions of influence scores for test pairs
- `test_influence_heatmap_X_Y.png` - Heatmaps showing which training samples influence each test pair
- `test_influence_top_bottom_X_Y.png` - Most and least influential training samples for each test pair
- `influence_by_angle_similarity.png` - Analysis of how angle differences correlate with influence

### 3. `generate_tracin_report.py`

Creates a comprehensive PDF report combining all TracIn visualizations with analysis and recommendations.

**Usage:**
```bash
python generate_tracin_report.py
```

**Output:**
- `tracin_reports/tracin_influence_report_YYYYMMDD_HHMMSS.pdf` - Complete PDF report with all visualizations and analysis

## Required Data

The visualization scripts expect TracIn influence data to be available at:

- Self-influence scores: `/Users/sbplab/Hank/sinetone_sliced/step_018_sliced/metadata/plastic_500hz_metadata.json`
- Test influence scores: `/Users/sbplab/Hank/sinetone_sliced/step_018_sliced/metadata/plastic_500hz_influence_metadata.json`

To generate this data, run the TracIn computation script with appropriate parameters:

```bash
# For self-influence scores
python compute_tracin_influence.py --frequency 500hz --material plastic \
    --checkpoint-dir saved_models/model_checkpoints/plastic_500hz_ghm_20250418_161140/ \
    --compute-self-influence --loss-type ghm

# For test influence scores
python compute_tracin_influence.py --frequency 500hz --material plastic \
    --checkpoint-dir saved_models/model_checkpoints/plastic_500hz_ghm_20250418_161140/ \
    --compute-influence --loss-type ghm
```

## Key Insights from TracIn Visualizations

The visualizations help identify:

1. The most difficult training samples (highest self-influence)
2. Which training samples are most helpful or harmful for specific test samples
3. Patterns in influence scores related to angle differences
4. Training samples that could be removed or augmented to improve model performance

## Dependencies

- matplotlib
- seaborn
- numpy
- matplotlib-backend-pdf
- other standard Python libraries

## Future Improvements

Potential enhancements to the visualization system:
- Interactive visualizations (e.g., with Plotly or Bokeh)
- Clustering analysis of influence patterns
- Anomaly detection for identifying unusual influence relationships 