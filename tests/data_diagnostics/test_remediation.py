import pytest
import torch
import numpy as np
import os
from utils.data_diagnostics.remediation import RemediationStrategies
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
def setup_remediation(tmp_path):
    """Fixture to create a detector, dataset, and remediation instance for testing"""
    model = SimpleModel()
    dataset = SimpleDataset()
    detector = ProblemSampleDetector(model, dataset)
    
    # Run detection to populate with results
    detector.detect_gradient_anomalies()
    detector.detect_feature_space_outliers()
    detector.detect_consistent_errors()
    
    # Create log directory
    log_dir = str(tmp_path / "remediation_logs")
    
    # Create remediation instance
    config = {"log_dir": log_dir}
    remediation = RemediationStrategies(detector, dataset, config)
    
    return model, dataset, detector, remediation, log_dir

def test_init(setup_remediation):
    """Test if the RemediationStrategies initializes correctly"""
    _, dataset, detector, remediation, log_dir = setup_remediation
    
    assert remediation.detector is detector
    assert remediation.dataset is dataset
    assert remediation.config is not None
    assert remediation.log_dir == log_dir
    assert os.path.exists(log_dir)
    assert hasattr(remediation, 'strategies')

def test_suggest_relabeling(setup_remediation):
    """Test if relabeling suggestions work"""
    _, _, _, remediation, _ = setup_remediation
    
    # Get relabeling suggestions
    suggestions = remediation.suggest_relabeling()
    
    # Verify results
    assert isinstance(suggestions, dict)
    assert 'samples' in suggestions
    assert 'confidence_scores' in suggestions
    assert len(suggestions['samples']) == len(suggestions['confidence_scores'])
    
    # Test with custom parameters
    suggestions = remediation.suggest_relabeling(confidence_threshold=0.7, max_samples=50)
    assert isinstance(suggestions, dict)
    assert 'samples' in suggestions
    assert len(suggestions['samples']) <= 50

def test_generate_sample_weights(setup_remediation):
    """Test if sample weight generation works"""
    _, _, _, remediation, _ = setup_remediation
    
    # Generate weights
    weights = remediation.generate_sample_weights()
    
    # Verify results
    assert isinstance(weights, (dict, np.ndarray))
    if isinstance(weights, dict):
        # Check dictionary format
        assert len(weights) > 0
    else:
        # Check array format
        assert len(weights) == len(remediation.dataset)
    
    # Test with different method
    weights = remediation.generate_sample_weights(method='influence_based', alpha=0.8)
    assert isinstance(weights, (dict, np.ndarray))

def test_suggest_augmentation_strategies(setup_remediation):
    """Test if augmentation strategy suggestions work"""
    _, _, _, remediation, _ = setup_remediation
    
    # Define a sample strategy pool
    strategy_pool = ['rotation', 'noise', 'flip']
    
    # Get augmentation strategy suggestions
    suggestions = remediation.suggest_augmentation_strategies(strategy_pool=strategy_pool)
    
    # Verify results
    assert isinstance(suggestions, dict)
    assert len(suggestions) > 0
    for sample_idx, strategies in suggestions.items():
        assert isinstance(strategies, list)
        # Strategies should be from the pool
        for strategy in strategies:
            assert strategy in strategy_pool

def test_generate_synthetic_samples(setup_remediation):
    """Test if synthetic sample generation works"""
    _, _, _, remediation, _ = setup_remediation
    
    # Generate synthetic samples
    samples, labels = remediation.generate_synthetic_samples()
    
    # Verify results
    assert isinstance(samples, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert samples.shape[0] == labels.shape[0]
    assert samples.shape[1:] == remediation.dataset.data.shape[1:]
    
    # Test with different parameters
    samples, labels = remediation.generate_synthetic_samples(
        method='smote', problem_class_indices=[0], n_samples=10
    )
    assert isinstance(samples, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert samples.shape[0] == labels.shape[0]

def test_apply_remediation(setup_remediation):
    """Test if remediation strategies can be applied"""
    _, dataset, _, remediation, _ = setup_remediation
    
    # Generate weights for weighted sampling
    weights = remediation.generate_sample_weights()
    
    # Apply remediation strategy
    remediated_dataset = remediation.apply_remediation(
        dataset, strategy="weighted_sampling", weights=weights
    )
    
    # Verify results
    assert remediated_dataset is not None
    assert isinstance(remediated_dataset, torch.utils.data.Dataset)
    assert len(remediated_dataset) > 0
    
    # Test with a different strategy
    remediated_dataset = remediation.apply_remediation(
        dataset, strategy="augmentation",
        augmentation_strategies={'rotation': 0.5, 'noise': 0.5}
    )
    assert isinstance(remediated_dataset, torch.utils.data.Dataset)

def test_evaluate_remediation_effect(setup_remediation):
    """Test if remediation effects can be evaluated"""
    model, dataset, _, remediation, _ = setup_remediation
    
    # Apply a simple remediation like weighted sampling
    weights = remediation.generate_sample_weights()
    remediated_dataset = remediation.apply_remediation(
        dataset, strategy="weighted_sampling", weights=weights
    )
    
    # Evaluate effect
    eval_results = remediation.evaluate_remediation_effect(model, remediated_dataset)
    
    # Verify results
    assert isinstance(eval_results, dict)
    assert 'original_metrics' in eval_results
    assert 'remediated_metrics' in eval_results
    assert 'improvement' in eval_results

def test_log_remediation_process(setup_remediation):
    """Test if remediation process can be logged"""
    _, _, _, remediation, log_dir = setup_remediation
    
    # Define strategy, params, and results
    strategy = "weighted_sampling"
    params = {"alpha": 0.8}
    results = {"accuracy": 0.85, "improvement": 0.05}
    
    # Log the process
    log_path = remediation.log_remediation_process(strategy, params, results)
    
    # Verify log file was created
    assert os.path.exists(log_path)
    assert log_path.startswith(log_dir) 