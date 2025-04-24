# Gradient Analysis Results

This directory contains experimental results from gradient analysis experiments.

## Frequency-Specific Analysis

The `frequency_analysis` directory contains results broken down by frequency:
- `500hz/`: Experimental results for 500Hz frequency
- `1000hz/`: Experimental results for 1000Hz frequency
- `3000hz/`: Experimental results for 3000Hz frequency

Each frequency directory contains epoch-by-epoch bin visualizations and heatmaps showing gradient behavior during training.

## GHM Analysis

The `ghm_analysis` directory contains results from experiments with Gradient Harmonizing Mechanism:
- `ghm/`: Standard GHM experimental results
- `ghm_alpha0.5_analysis/`: Analysis of GHM with alpha=0.5 parameter
- `ghm_analysis/`: General GHM analysis results
- `ghm_vs_standard/`: Comparative analysis between GHM and standard training

These results demonstrate how GHM affects gradient behavior and training dynamics compared to standard optimization methods. 