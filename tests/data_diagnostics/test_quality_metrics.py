import pytest
import torch
import numpy as np
from utils.data_diagnostics import quality_metrics

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
def setup_data():
    """Fixture to create a model, sample, target, and dataset for testing"""
    model = SimpleModel()
    dataset = SimpleDataset()
    sample_idx = 0  # Use the first sample for testing
    sample = dataset.data[sample_idx]
    target = dataset.labels[sample_idx]
    
    return model, sample, target, dataset

def test_calculate_sample_difficulty(setup_data):
    """Test the sample difficulty calculation function"""
    model, sample, target, _ = setup_data
    
    # Test with default parameters
    difficulty = quality_metrics.calculate_sample_difficulty(model, sample, target)
    assert isinstance(difficulty, float)
    assert 0 <= difficulty <= 1  # Difficulty should be normalized between 0 and 1
    
    # Test with custom topk
    difficulty = quality_metrics.calculate_sample_difficulty(model, sample, target, topk=(1, 3))
    assert isinstance(difficulty, float)
    assert 0 <= difficulty <= 1

def test_calculate_sample_influence(setup_data):
    """Test the sample influence calculation function"""
    model, sample, target, dataset = setup_data
    
    # Define simple criterion for testing
    criterion = torch.nn.CrossEntropyLoss()
    
    # Create a small validation set for testing
    validation_indices = torch.randperm(len(dataset))[:10]
    validation_set = [dataset[i] for i in validation_indices]
    
    # Test the function
    influence = quality_metrics.calculate_sample_influence(model, sample, target, validation_set, criterion)
    assert isinstance(influence, float)
    # Influence can be positive or negative

def test_calculate_feature_space_density(setup_data):
    """Test the feature space density calculation function"""
    model, sample, _, dataset = setup_data
    
    # Test with default parameters
    density = quality_metrics.calculate_feature_space_density(model, sample, dataset)
    assert isinstance(density, float)
    assert density >= 0  # Density should be non-negative
    
    # Test with different method
    density = quality_metrics.calculate_feature_space_density(model, sample, dataset, method='knn', k=3)
    assert isinstance(density, float)
    assert density >= 0

def test_calculate_prediction_stability(setup_data):
    """Test the prediction stability calculation function"""
    model, sample, _, _ = setup_data
    
    # Define simple augmentation function
    def simple_augmentation(x):
        return x + torch.randn_like(x) * 0.1
    
    # Test with default parameters
    stability = quality_metrics.calculate_prediction_stability(model, sample)
    assert isinstance(stability, float)
    assert 0 <= stability <= 1  # Stability should be normalized between 0 and 1
    
    # Test with custom augmentations
    stability = quality_metrics.calculate_prediction_stability(
        model, sample, augmentations=[simple_augmentation], n_augmentations=5
    )
    assert isinstance(stability, float)
    assert 0 <= stability <= 1

def test_calculate_loss_landscape(setup_data):
    """Test the loss landscape calculation function"""
    model, sample, target, _ = setup_data
    
    # Test with default parameters
    landscape = quality_metrics.calculate_loss_landscape(model, sample, target)
    assert isinstance(landscape, dict)
    assert 'loss_values' in landscape
    assert 'curvature' in landscape
    
    # Test with custom parameters
    landscape = quality_metrics.calculate_loss_landscape(model, sample, target, epsilon=0.05, n_points=5)
    assert isinstance(landscape, dict)
    assert 'loss_values' in landscape
    assert 'curvature' in landscape

def test_calculate_comprehensive_quality_score(setup_data):
    """Test the comprehensive quality score calculation function"""
    model, sample, target, dataset = setup_data
    
    # Generate individual metrics
    metrics_dict = {
        'difficulty': quality_metrics.calculate_sample_difficulty(model, sample, target),
        'density': quality_metrics.calculate_feature_space_density(model, sample, dataset),
        'stability': quality_metrics.calculate_prediction_stability(model, sample)
    }
    
    # Test with default weights
    score = quality_metrics.calculate_comprehensive_quality_score(metrics_dict)
    assert isinstance(score, float)
    
    # Test with custom weights
    weights = {'difficulty': 0.5, 'density': 0.3, 'stability': 0.2}
    score = quality_metrics.calculate_comprehensive_quality_score(metrics_dict, weights=weights)
    assert isinstance(score, float) 