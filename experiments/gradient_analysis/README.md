# Gradient Analysis Experiments

This directory contains experiments and analyses related to gradient behavior during model training.

## Directory Structure

- `visualizations/`: Gradient visualization plots and heatmaps
  - `standard/`: Standard gradient descent visualizations
  - `svrg/`: SVRG (Stochastic Variance Reduced Gradient) visualizations
  - `comparisons/`: Comparison visualizations between different methods
  
- `results/`: Experimental results data
  - `ghm_analysis/`: Results from GHM (Gradient Harmonizing Mechanism) experiments
  - `frequency_analysis/`: Results broken down by frequency (500Hz, 1000Hz, 3000Hz)

- `docs/`: Documentation and analysis notes

## Experiment Overview

This collection includes visualizations and results from experiments analyzing gradient behavior during training, including:

1. Gradient statistics for standard SGD vs SVRG
2. Frequency-specific gradient heatmaps (500Hz, 1000Hz, 3000Hz)
3. GHM analysis across different frequencies
4. Epoch-by-epoch comparison of gradient behavior

The experiments focus on understanding training dynamics and optimizing the learning process for audio angle classification tasks. 