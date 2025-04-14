# This file was created during the refactoring process

"""
Model Analysis Submodule

This module provides tools for analyzing model structure and evaluating
model performance through confusion matrices and other metrics.
"""

from utils.analysis.model.confusion_analyzer import (
    analyze_confusion_matrix,
    analyze_angle_predictions,
    generate_conf_matrix,
    plot_conf_matrix,
    load_model,
    prepare_data_loader,
    compute_angle_accuracy
)

from utils.analysis.model.structure_analyzer import (
    analyze_model_structure,
    check_model_structure,
    format_model_info
)

__all__ = [
    # Confusion Matrix Analyzer
    'analyze_confusion_matrix',
    'analyze_angle_predictions',
    'generate_conf_matrix',
    'plot_conf_matrix',
    'load_model',
    'prepare_data_loader',
    'compute_angle_accuracy',
    
    # Model Structure Analyzer
    'analyze_model_structure',
    'check_model_structure',
    'format_model_info'
]