import pytest
import torch
import numpy as np
from utils.data_diagnostics.sample_detector import ProblemSampleDetector

class SimpleModel(torch.nn.Module):
    """A simple model for testing purposes"""
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)
    
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
def setup_detector():
    """Fixture to create a model, dataset, and detector for testing"""
    model = SimpleModel()
    dataset = SimpleDataset()
    detector = ProblemSampleDetector(model, dataset)
    return model, dataset, detector

def test_init(setup_detector):
    """Test if the ProblemSampleDetector initializes correctly"""
    model, dataset, detector = setup_detector
    
    assert detector.model is model
    assert detector.dataset is dataset
    assert hasattr(detector, 'config')
    assert hasattr(detector, 'device')
    assert detector.results == {}

def test_detect_gradient_anomalies(setup_detector):
    """Test if the gradient anomaly detection works"""
    _, _, detector = setup_detector
    
    # Run detection
    results = detector.detect_gradient_anomalies()
    
    # Verify results format
    assert isinstance(results, dict)
    assert 'anomaly_indices' in results
    assert 'gradient_values' in results
    assert len(results['anomaly_indices']) == len(results['gradient_values'])
    
    # Check that results are stored
    assert 'gradient_anomalies' in detector.results

def test_detect_feature_space_outliers(setup_detector):
    """Test if the feature space outlier detection works"""
    _, _, detector = setup_detector
    
    # Run detection
    results = detector.detect_feature_space_outliers()
    
    # Verify results format
    assert isinstance(results, dict)
    assert 'outlier_indices' in results
    assert 'outlier_scores' in results
    assert len(results['outlier_indices']) == len(results['outlier_scores'])
    
    # Check that results are stored
    assert 'feature_space_outliers' in detector.results

def test_detect_consistent_errors(setup_detector):
    """Test if the consistent error detection works"""
    _, _, detector = setup_detector
    
    # Run detection with small n_folds to make it faster for tests
    results = detector.detect_consistent_errors(n_folds=3)
    
    # Verify results format
    assert isinstance(results, dict)
    assert 'error_indices' in results
    assert 'error_rates' in results
    assert len(results['error_indices']) == len(results['error_rates'])
    
    # Check that results are stored
    assert 'consistent_errors' in detector.results

def test_comprehensive_detection(setup_detector):
    """Test if the comprehensive detection works"""
    _, _, detector = setup_detector
    
    # Run comprehensive detection
    results = detector.run_comprehensive_detection()
    
    # Verify results format
    assert isinstance(results, dict)
    assert 'gradient_anomalies' in results
    assert 'feature_space_outliers' in results
    assert 'consistent_errors' in results
    
    # Check problem samples ranking
    ranking = detector.get_problem_samples_ranking()
    assert isinstance(ranking, list)
    assert len(ranking) > 0

def test_save_and_load_results(setup_detector, tmp_path):
    """Test if the results can be saved and loaded"""
    _, _, detector = setup_detector
    
    # Generate some results
    detector.run_comprehensive_detection()
    
    # Save results
    filepath = tmp_path / "test_results.json"
    detector.save_results(filepath)
    
    # Create a new detector
    new_model = SimpleModel()
    new_dataset = SimpleDataset()
    new_detector = ProblemSampleDetector(new_model, new_dataset)
    
    # Load results
    new_detector.load_results(filepath)
    
    # Verify that the results were loaded correctly
    assert new_detector.results.keys() == detector.results.keys()
    for key in detector.results:
        if isinstance(detector.results[key], dict):
            assert detector.results[key].keys() == new_detector.results[key].keys() 