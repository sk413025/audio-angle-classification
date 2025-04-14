# This file was created during the refactoring process

"""
GHM Training and Analysis Module

This module provides tools for analyzing and visualizing GHM (Gradient Harmonized Mechanism)
training results and model performance.
"""

# GHM Analysis modules
from utils.analysis.ghm.results_analyzer import (
    analyze_ghm_results,
    find_latest_training_dirs,
    load_training_history,
    plot_training_comparison,
    analyze_single_run,
    plot_confusion_matrix,
    create_parameter_impact_plot,
    create_frequency_comparison_plot
)

from utils.analysis.ghm.detailed_analyzer import (
    analyze_ghm_details,
    visualize_bin_evolution,
    analyze_gradient_statistics,
    compare_different_ghm_params,
    get_stats_files,
    load_ghm_stats
)

# Model Analysis modules
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
    # GHM Results Analyzer
    'analyze_ghm_results',
    'find_latest_training_dirs',
    'load_training_history',
    'plot_training_comparison',
    'analyze_single_run',
    'plot_confusion_matrix',
    'create_parameter_impact_plot',
    'create_frequency_comparison_plot',
    
    # GHM Detailed Analyzer
    'analyze_ghm_details',
    'visualize_bin_evolution',
    'analyze_gradient_statistics',
    'compare_different_ghm_params',
    'get_stats_files',
    'load_ghm_stats',
    
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