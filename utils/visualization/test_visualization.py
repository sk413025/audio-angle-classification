"""
Test script for visualization module.

This script tests the functionality of the visualization module 
by creating sample data and visualizing it.
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime

# Add root to path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from losses.ghm_loss import GHMRankingLoss
from utils.visualization import plot_training_history, plot_ghm_statistics, plot_gradient_distribution, set_plot_style

def create_sample_history():
    """Create a sample training history for testing visualization."""
    epochs = 30
    
    # Create a sample history dictionary
    history = {
        'epoch': list(range(1, epochs + 1)),
        'train_loss_main': [1.0 - i * 0.03 + 0.001 * np.random.rand() for i in range(epochs)],
        'val_loss_main': [1.05 - i * 0.025 + 0.005 * np.random.rand() for i in range(epochs)],
        'train_accuracy': [60 + i * 1.2 + 0.5 * np.random.rand() for i in range(epochs)],
        'val_accuracy': [58 + i * 1.1 + 0.7 * np.random.rand() for i in range(epochs)],
        'train_loss_standard_log': [1.1 - i * 0.028 + 0.002 * np.random.rand() for i in range(epochs)],
        'val_loss_standard_log': [1.15 - i * 0.022 + 0.006 * np.random.rand() for i in range(epochs)],
        'args': {
            'loss_type': 'ghm',
            'frequency': '1000hz',
            'material': 'plastic'
        }
    }
    
    return history

def create_sample_ghm_outputs():
    """Create sample GHM outputs for testing visualization."""
    # Create a simple GHMRankingLoss instance
    criterion = GHMRankingLoss(margin=1.0, bins=10, alpha=0.75)
    
    # Create sample outputs and targets
    batch_size = 16
    outputs1 = torch.randn(batch_size)
    outputs2 = torch.randn(batch_size)
    targets = torch.sign(outputs1 - outputs2 + 0.1 * torch.randn(batch_size))
    
    # Initialize edges and distributions (normally done by criterion during forward pass)
    criterion.edges = torch.linspace(0, 10, 11)
    criterion.distribution = torch.zeros(10)
    criterion.distribution[0:5] = torch.tensor([8, 5, 3, 2, 1])
    
    return criterion, outputs1, outputs2, targets

def main():
    """Test visualization module functionality."""
    print("Testing visualization module...")
    
    # Set a consistent style
    set_plot_style('default')
    
    # Create test output directory
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_output')
    os.makedirs(test_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Test training history visualization
    print("\nTesting training history visualization...")
    history = create_sample_history()
    history_plot_path = os.path.join(test_dir, f'test_history_plot_{timestamp}.png')
    plot_training_history(history, history_plot_path)
    print(f"  Plot saved to: {history_plot_path}")
    
    # Test GHM visualization
    print("\nTesting GHM visualization...")
    criterion, outputs1, outputs2, targets = create_sample_ghm_outputs()
    ghm_stats_path = os.path.join(test_dir, f'test_ghm_stats_{timestamp}')
    os.makedirs(ghm_stats_path, exist_ok=True)
    
    # Test direct gradient distribution plotting
    grad_dist_path = os.path.join(test_dir, f'grad_dist_{timestamp}')
    plot_gradient_distribution(criterion, test_dir, grad_dist_path)
    print(f"  Gradient distribution plot saved to: {os.path.join(test_dir, f'{grad_dist_path}_gradient_dist.png')}")
    
    # Test GHM statistics plotting
    plot_ghm_statistics(criterion, outputs1, outputs2, targets, test_dir, f'ghm_stats_{timestamp}')
    print(f"  GHM statistics visualized in: {test_dir}")
    
    print("\nAll visualization tests completed successfully!")

if __name__ == "__main__":
    main() 