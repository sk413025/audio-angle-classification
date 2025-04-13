# Agent Instructions: Automated GHM Training and Analysis

## 1. Purpose

This document provides instructions for an automated agent to execute training runs using the Gradient Harmonizing Mechanism (GHM) loss function within the `angle_classification_deg6` project and subsequently locate the generated analysis outputs (plots and statistics).

## 2. Prerequisites

*   Access to the project workspace (`angle_classification_deg6`).
*   Ability to execute shell commands (specifically `python`).
*   Ability to read standard output (stdout) from executed commands.
*   Ability to list directory contents and identify files.

## 3. Step 1: Execute GHM Training

The primary script for training is `train.py`. To run with GHM enabled, use the `--loss-type ghm` argument (this is also the default).

**Command Template:**

```bash
python train.py --frequency <frequency_value> --loss-type ghm [additional_options] | cat
```

**Key Arguments:**

*   `--frequency <frequency_value>`: **(Required)** The frequency data to use. Must be one of: `'500hz'`, `'1000hz'`, `'3000hz'`. Cannot be `'all'` for a single automated run intended for analysis.
*   `--loss-type ghm`: **(Required for clarity, although default)** Ensures GHM loss is used.
*   `--epochs <N>`: (Optional) Number of epochs to train (e.g., `--epochs 10`). Default is 30. Use a small number (e.g., 1 or 2) for quick test runs.
*   `--seed <N>`: (Optional) Specify a random seed for reproducibility (e.g., `--seed 42`). Default is from `config.py` or 42. Using a fixed seed is highly recommended for comparable results.
*   `--ghm-bins <N>`: (Optional) Number of bins for GHM loss (default: 10).
*   `--ghm-alpha <F>`: (Optional) Alpha parameter for GHM loss (default: 0.75).
*   `--margin <F>`: (Optional) Margin for the ranking loss (default: from `config.py` or 1.0).
*   Other arguments (e.g., `--learning-rate`, `--batch-size`) can be added to override defaults from `config.py` if needed.

**Example Command (1000Hz, 2 epochs, seed 42):**

```bash
python train.py --frequency 1000hz --loss-type ghm --epochs 2 --seed 42 | cat
```

**Execution and Output Capture:**

*   Execute the chosen command.
*   **Crucially, capture the complete standard output (stdout) of the `train.py` script.** This output contains the exact paths where the results and analysis files are saved.

## 4. Step 2: Identify Output Paths from Script Log

Parse the captured stdout from Step 1 to find the following lines (the timestamp and exact paths will vary):

*   `Model checkpoint saved to: <path_to_checkpoint_dir>/model_epoch_<N>.pt`
*   `Training history plot saved to: <path_to_plots_dir>/training_history_<timestamp>.png`
*   `Training history saved to: <path_to_checkpoint_dir>/training_history_<timestamp>.pkl`
*   **(If GHM was used)** Lines indicating where GHM statistics were saved (e.g., `calculate_ghm_statistics` output, often pointing towards `<path_to_stats_dir>`).
*   **(If GHM was used)** Lines indicating where GHM gradient distribution plots were saved (e.g., `plot_gradient_distribution` output, often pointing towards `<path_to_plots_dir>`).

Extract the base directories for checkpoints, plots, and stats:
*   `<path_to_checkpoint_dir>` (e.g., `/path/to/project/saved_models/model_checkpoints/plastic_1000hz_ghm_20250413_224252`)
*   `<path_to_plots_dir>` (e.g., `/path/to/project/saved_models/plots/plastic_1000hz_ghm_20250413_224252`)
*   `<path_to_stats_dir>` (e.g., `/path/to/project/saved_models/stats/plastic_1000hz_ghm_20250413_224252`) - This directory only exists if `loss-type` was `ghm`.

## 5. Step 3: Locate GHM Analysis Artifacts

Using the paths identified in Step 2:

1.  **Locate Gradient Distribution Plots:**
    *   List the contents of the `<path_to_plots_dir>`.
    *   Identify files matching the pattern `grad_dist_epoch*_batch*_<timestamp>.png`. These plots visualize the gradient norm distribution before and after GHM weighting for specific batches.
    *   **Action:** Report the full paths of all found gradient distribution `.png` files.

2.  **Locate GHM Statistics Files:**
    *   List the contents of the `<path_to_stats_dir>`. (Skip if this directory doesn't exist).
    *   Identify files matching the pattern `epoch*_batch*_<timestamp>_ghm_stats.pkl`. These files contain detailed numerical statistics (gradient norms, GHM weights, bin counts) that require Python's `pickle` module to load and inspect.
    *   **Action:** Report the full paths of all found GHM statistics `.pkl` files.

3.  **Locate General Training History Plot:**
    *   List the contents of the `<path_to_plots_dir>`.
    *   Identify the file matching `training_history_<timestamp>.png`. This plot shows the overall training/validation loss and accuracy curves.
    *   **Action:** Report the full path of the found training history `.png` file.

## 6. Step 4: Report Results

Provide a report summarizing the execution, including:

*   Confirmation that the training command (specify the command used) completed.
*   The identified paths for:
    *   Checkpoint directory.
    *   Plots directory.
    *   Stats directory (if applicable).
*   A list of the full paths for:
    *   Located GHM gradient distribution plots (`.png`).
    *   Located GHM statistics files (`.pkl`).
    *   The main training history plot (`.png`).
*   Report any errors encountered during training execution (check stderr or specific error messages in stdout).

## 7. Important Notes

*   Always use the `| cat` pipe when executing `train.py` in an automated environment to prevent potential blocking issues with buffered output.
*   Rely on parsing the script's stdout to find output paths rather than trying to predict them, as timestamps will differ.
*   Ensure the specified frequency exists in the dataset for the chosen material to avoid data loading errors.
*   If the training script exits with an error, report the error instead of proceeding to locate outputs. 