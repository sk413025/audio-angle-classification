import pytest
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from utils.data_diagnostics.diagnostics_visualizer import DiagnosticsVisualizer
from utils.data_diagnostics.sample_detector import ProblemSampleDetector

class SimpleModel(torch.nn.Module):
    """A simple model for testing purposes"""
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 3)
    
    def forward(self, x):
        return self.linear(x)

class SimpleDataset(torch.utils.data.Dataset):
    """A simple dataset for testing purposes"""
    def __init__(self, n_samples=100):
        self.data = torch.randn(n_samples, 10)
        # Create controlled labels for testing (0, 1, 2)
        self.labels = torch.randint(0, 3, (n_samples,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

@pytest.fixture
def setup_visualizer(tmp_path):
    """Fixture to create a detector and visualizer for testing"""
    model = SimpleModel()
    dataset = SimpleDataset()
    detector = ProblemSampleDetector(model, dataset)
    
    # Run some detection methods to populate the detector with results
    detector.detect_gradient_anomalies()
    detector.detect_feature_space_outliers()
    detector.detect_consistent_errors()
    
    # Create output directory for visualizations
    output_dir = str(tmp_path / "visualization_test")
    
    # Create visualizer with custom output directory
    config = {"output_dir": output_dir}
    visualizer = DiagnosticsVisualizer(detector, dataset, config)
    
    return model, dataset, detector, visualizer, output_dir

def test_init(setup_visualizer):
    """Test if the DiagnosticsVisualizer initializes correctly"""
    _, dataset, detector, visualizer, output_dir = setup_visualizer
    
    assert visualizer.detector is detector
    assert visualizer.dataset is dataset
    assert visualizer.config is not None
    assert visualizer.output_dir == output_dir
    assert os.path.exists(output_dir)

def test_visualize_feature_space(setup_visualizer):
    """Test if the feature space visualization works"""
    _, _, _, visualizer, _ = setup_visualizer
    
    # Generate visualization with default parameters
    fig = visualizer.visualize_feature_space()
    
    # Verify result
    assert isinstance(fig, plt.Figure)
    
    # Test with different parameters
    fig = visualizer.visualize_feature_space(method='pca', dims=2)
    assert isinstance(fig, plt.Figure)

def test_visualize_gradient_distribution(setup_visualizer):
    """Test if the gradient distribution visualization works"""
    _, _, _, visualizer, _ = setup_visualizer
    
    # Generate visualization
    fig = visualizer.visualize_gradient_distribution()
    
    # Verify result
    assert isinstance(fig, plt.Figure)
    
    # Test with custom parameters
    fig = visualizer.visualize_gradient_distribution(threshold=0.9, bins=30)
    assert isinstance(fig, plt.Figure)

def test_visualize_error_patterns(setup_visualizer):
    """Test if the error patterns visualization works"""
    _, _, _, visualizer, _ = setup_visualizer
    
    # Generate visualization
    fig = visualizer.visualize_error_patterns()
    
    # Verify result
    assert isinstance(fig, plt.Figure)
    
    # Test with custom parameters
    fig = visualizer.visualize_error_patterns(top_n=5)
    assert isinstance(fig, plt.Figure)

def test_visualize_quality_metrics(setup_visualizer):
    """Test if the quality metrics visualization works"""
    _, _, _, visualizer, _ = setup_visualizer
    
    # Generate some quality metrics data
    metrics_data = {
        'sample_0': {'difficulty': 0.7, 'density': 0.3, 'stability': 0.9},
        'sample_1': {'difficulty': 0.2, 'density': 0.8, 'stability': 0.5},
        'sample_2': {'difficulty': 0.5, 'density': 0.5, 'stability': 0.5}
    }
    
    # Generate visualization
    fig = visualizer.visualize_quality_metrics(metrics_data)
    
    # Verify result
    assert isinstance(fig, plt.Figure)

def test_save_visualization(setup_visualizer):
    """Test if visualizations can be saved to file"""
    _, _, _, visualizer, output_dir = setup_visualizer
    
    # Generate a simple matplotlib figure
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    
    # Save the figure
    filename = "test_figure.png"
    filepath = visualizer.save_visualization(fig, filename)
    
    # Verify the file was saved
    assert os.path.exists(filepath)
    assert filepath == os.path.join(output_dir, filename)

def test_generate_interactive_visualization(setup_visualizer):
    """Test if interactive visualizations can be generated"""
    _, _, _, visualizer, output_dir = setup_visualizer
    
    # Generate interactive visualization
    result = visualizer.generate_interactive_visualization()
    
    # Verify result (either a file path or a plotly/bokeh figure)
    assert result is not None
    if isinstance(result, str):
        assert os.path.exists(result)

def test_generate_comprehensive_report(setup_visualizer):
    """Test if a comprehensive report can be generated"""
    _, _, _, visualizer, output_dir = setup_visualizer
    
    # Generate report
    filename = "test_report.html"
    report_path = visualizer.generate_comprehensive_report(filename)
    
    # Verify report was created
    assert os.path.exists(report_path)
    assert report_path == os.path.join(output_dir, filename) 