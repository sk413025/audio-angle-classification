"""
Problem Data Identification and Diagnostic Module.

This module provides tools for identifying and diagnosing problematic data samples
in machine learning datasets. It helps find anomalous or low-quality samples that
may affect model performance, and provides visualizations and remediation strategies.

Main components:
- sample_detector: For detecting problem samples using various methods
- quality_metrics: For evaluating sample quality through various metrics
- diagnostics_visualizer: For visualizing problem data characteristics
- remediation: For implementing data remediation strategies
"""

from utils.data_diagnostics.sample_detector import ProblemSampleDetector
from utils.data_diagnostics.quality_metrics import (
    calculate_sample_difficulty,
    calculate_sample_influence,
    calculate_feature_space_density,
    calculate_prediction_stability,
    calculate_loss_landscape,
    calculate_comprehensive_quality_score
)
from utils.data_diagnostics.diagnostics_visualizer import DiagnosticsVisualizer
from utils.data_diagnostics.remediation import RemediationStrategies

__all__ = [
    # Sample detector
    'ProblemSampleDetector',
    
    # Quality metrics
    'calculate_sample_difficulty',
    'calculate_sample_influence',
    'calculate_feature_space_density',
    'calculate_prediction_stability',
    'calculate_loss_landscape',
    'calculate_comprehensive_quality_score',
    
    # Visualizer
    'DiagnosticsVisualizer',
    
    # Remediation
    'RemediationStrategies'
] 