# GHM Training and Analysis Refactoring

This project has undergone a refactoring to improve code organization, maintainability, and extensibility, with a focus on the GHM training analysis code.

## Directory Structure Changes

The following directory structure changes were made:

```
audio-angle-classification/
├── utils/
│   ├── analysis/                       # New analysis module
│   │   ├── ghm/                        # GHM-specific analysis
│   │   │   ├── detailed_analyzer.py    # Detailed GHM analysis
│   │   │   └── results_analyzer.py     # GHM results analysis
│   │   └── model/                      # Model analysis
│   │       ├── confusion_analyzer.py   # Confusion matrix analysis
│   │       └── structure_analyzer.py   # Model structure analysis
├── logs/                               # New logs directory
│   └── training/                       # Training logs
│       ├── ghm/                        # GHM training logs
│       │   ├── 1000hz/
│       │   ├── 500hz/
│       │   └── 3000hz/
│       └── standard/                   # Standard training logs
├── reports/                            # Reports directory
│   └── ghm_training/                   # GHM training reports
│       └── ghm_training_report.md
├── results/                            # Analysis results directory
│   ├── ghm/                            # GHM analysis results
│   └── confusion_matrix/               # Confusion matrix results
├── scripts/                            # New scripts directory
│   ├── analyze_ghm.py                  # GHM analysis script
│   ├── analyze_confusion.py            # Confusion matrix analysis script
│   └── check_model.py                  # Model structure check script
└── tests/                              # New tests directory
    ├── test_imports.py                 # Import tests
    ├── test_ghm_analysis.py            # GHM analysis tests
    └── integration/                    # Integration tests
        └── test_analysis_pipeline.py   # Pipeline tests
```

## Refactored Modules

The following modules were created or updated:

### Analysis Modules

- **utils/analysis/ghm/results_analyzer.py**: Refactored from `analyze_ghm_results.py`
- **utils/analysis/ghm/detailed_analyzer.py**: Refactored from `analyze_ghm_details.py`
- **utils/analysis/model/confusion_analyzer.py**: Refactored from `confusion_matrix_analysis.py`
- **utils/analysis/model/structure_analyzer.py**: Refactored from `check_model_structure.py`

### Scripts

- **scripts/analyze_ghm.py**: Analyzes GHM training results
- **scripts/analyze_confusion.py**: Analyzes model performance via confusion matrices
- **scripts/check_model.py**: Checks the structure of a saved model

### Tests

- **tests/test_imports.py**: Tests that imports work correctly
- **tests/test_ghm_analysis.py**: Tests GHM analysis functionality
- **tests/integration/test_analysis_pipeline.py**: Tests the analysis pipeline end-to-end

## File Organizations

- Log files have been moved to the `logs/training/` directory
- The GHM training report has been moved to `reports/ghm_training/`
- Analysis results will be saved to the `results/` directory

## Usage Examples

### Analyzing GHM Results

```bash
python scripts/analyze_ghm.py --base-dir saved_models --output-dir results/ghm --frequency 1000hz
```

### Analyzing Model Performance

```bash
python scripts/analyze_confusion.py --model-path saved_models/model_best.pt --frequency 1000hz
```

### Checking Model Structure

```bash
python scripts/check_model.py --model-path saved_models/model_best.pt
```

## Benefits of Refactoring

1. **Improved Modularity**: Code is organized into logical modules
2. **Better Maintainability**: Separating concerns makes code easier to maintain
3. **Easier Testing**: Modular code enables unit testing
4. **Enhanced Extensibility**: New analysis methods can be added easily
5. **Better Organization**: Files are stored in logical directories

## Running Tests

```bash
# Run all tests
python -m unittest discover

# Run specific test modules
python -m tests.test_imports
python -m tests.test_ghm_analysis
python -m tests.integration.test_analysis_pipeline
```

## Removed Files

The following files are no longer needed and can be removed after verifying the refactored code works correctly:

- `analyze_ghm_results.py`
- `analyze_ghm_details.py`
- `confusion_matrix_analysis.py`
- `check_model_structure.py`