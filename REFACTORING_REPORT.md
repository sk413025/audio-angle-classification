# Refactoring Report: Modular GHM Training

## 1. Background and Motivation

The original project structure integrated the GHM (Gradient Harmonizing Mechanism) loss and its associated utilities directly within the main training script (`train_with_ghm.py`). While functional, this approach had several drawbacks:

*   **Lack of Modularity**: GHM was tightly coupled with the core training logic, making it difficult to switch to standard training or experiment with other loss functions without significant code changes.
*   **Code Organization**: Utility functions (general, GHM-specific, debugging) and model definitions were mixed within the main script or scattered, hindering maintainability and readability.
*   **Flexibility**: The script relied on interactive prompts (`input()`) for configuration, which is less suitable for automated experiments or batch processing.

The primary motivation for this refactoring was to **improve the project's structure, making GHM an optional, modular component** within a more organized and flexible training framework.

## 2. Objective

The main goals of the refactoring were:

*   **Separate Concerns**: Isolate GHM-specific logic (loss function, utilities) into dedicated modules.
*   **Create Core Training Script**: Develop a central `train.py` script capable of handling standard training and optionally enabling GHM via configuration.
*   **Improve Organization**: Group related code into logical directories (`models/`, `losses/`, `utils/`).
*   **Enhance Flexibility**: Replace interactive prompts with command-line argument parsing (`argparse`) for easier configuration and automation.
*   **Maintain Functionality**: Ensure the refactored code produces results consistent with the original script when using the same parameters and seed.

## 3. Refactoring Steps Performed

The refactoring was performed incrementally to minimize risk and allow for verification at each stage:

1.  **Isolate GHM Loss Function**:
    *   Created `losses/` directory and `losses/__init__.py`.
    *   Moved `ghm_loss.py` into `losses/`.
    *   Updated import in `train_with_ghm.py` to `from losses.ghm_loss import GHMRankingLoss`.
    *   *Verified via test run.*

2.  **Isolate GHM Utility Functions**:
    *   Created `utils/` directory and `utils/__init__.py`.
    *   Moved `ghm_utils.py` into `utils/`.
    *   Updated import in `train_with_ghm.py` to `from utils import ghm_utils`.
    *   *Verified via test run.*

3.  **Isolate General Utility Functions**:
    *   Created `utils/common_utils.py`.
    *   Moved `worker_init_fn` and `set_seed` from `train_with_ghm.py` to `utils/common_utils.py`.
    *   Updated `train_with_ghm.py` to import from `utils.common_utils` and removed original definitions.
    *   *Verified via test run.*

4.  **Refactor Training Main Logic (Create `train.py`)**:
    *   Created `train.py`.
    *   Added necessary imports.
    *   Implemented `parse_arguments()` using `argparse`.
    *   Copied `train_model_with_ghm` logic into `train_model(args)` in `train.py`.
    *   Refactored `train_model` to accept `args`, select loss function (`standard` or `ghm`) based on `args.loss_type`, and conditionally call GHM utils.
    *   Adjusted logging, plotting, and saving paths to be more generic and reflect `args`.
    *   Implemented `main()` function in `train.py` to parse args and call `train_model`.
    *   Added `if __name__ == "__main__":` block.
    *   *Verified via test runs for both GHM and standard loss.*

5.  **Move Model File**:
    *   Created `models/` directory and `models/__init__.py`.
    *   Moved `simple_cnn_models_native.py` to `models/resnet_ranker.py`.
    *   Updated model import in `train.py`.
    *   *Verified via test run.*

6.  **Move Debugging Utilities**:
    *   Created `utils/debugging_utils.py`.
    *   Moved `verify_batch_consistency` function from `train_with_ghm.py` to `utils/debugging_utils.py`.
    *   Updated `train.py` to import and conditionally use `verify_batch_consistency` based on the `--verify-consistency` flag.
    *   *Verified via test run using the flag.*

7.  **Cleanup**:
    *   Removed the redundant `train_model_with_ghm` and `main` functions from `train_with_ghm.py`.
    *   Updated docstrings in `train_with_ghm.py` to mark it as deprecated and note issues with the remaining `check_batch_consistency` function.

## 4. How to Run GHM Training and Analysis (Post-Refactoring)

All training, including GHM-specific runs, should now be initiated using the new `train.py` script via the command line.

**Executing GHM Training:**

To run training using the GHM loss, ensure you specify `--loss-type ghm` (which is also the default). You can configure GHM parameters using the corresponding arguments.

```bash
python train.py --frequency <freq> --loss-type ghm [options]
```

**Example:** Train with GHM on 1000Hz data for 50 epochs, using specific GHM parameters and a custom seed:

```bash
python train.py --frequency 1000hz --loss-type ghm --epochs 50 --ghm-bins 15 --ghm-alpha 0.8 --margin 12.0 --seed 123
```

**Required Arguments:**

*   `--frequency <freq>`: Specify the frequency ('500hz', '1000hz', '3000hz', or 'all').

**Key Optional Arguments for GHM:**

*   `--loss-type ghm`: Explicitly select GHM loss (optional, as it's the default).
*   `--epochs <N>`: Number of training epochs (default: 30).
*   `--ghm-bins <N>`: Number of bins for GHM (default: 10).
*   `--ghm-alpha <F>`: Alpha value for GHM (default: 0.75).
*   `--margin <F>`: Margin value for the loss (default: from `config.py` or 1.0).
*   `--seed <N>`: Random seed (default: from `config.py` or 42).
*   `--learning-rate <F>`: Override learning rate (default: from `config.py`).
*   `--batch-size <N>`: Override batch size (default: from `config.py`).
*   `--checkpoint-interval <N>`: How often to save checkpoints (default: 5).

**Accessing GHM Visual Analysis:**

When training with `--loss-type ghm`, the script automatically performs and saves GHM-related analyses:

*   **Gradient Distribution Plots**: Plots showing the distribution of gradients (before and after GHM weighting) are saved for the first and last batch of each epoch. These are saved within the `saved_models/plots/<run_name>/` directory (where `<run_name>` includes material, frequency, loss type, and timestamp). Look for files named like `grad_dist_epoch<N>_batch<M>_<timestamp>.png`.
*   **GHM Statistics**: Detailed statistics about gradient norms, GHM weights, and effective number of examples per bin are calculated and saved as pickle files (`.pkl`) for the first and last batch of each epoch. These are saved within the `saved_models/stats/<run_name>/` directory. Look for files named like `epoch<N>_batch<M>_<timestamp>_ghm_stats.pkl`.

You can load the `.pkl` files using Python's `pickle` module to analyze the numerical statistics further if needed. The gradient distribution plots provide a direct visual assessment of how GHM is affecting the gradient flow during training.

This refactoring provides a more robust and flexible foundation for future experiments and development. 