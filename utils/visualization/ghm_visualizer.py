"""
Gradient Harmonizing Mechanism (GHM) visualization utilities.

This module provides functions for visualizing GHM-related statistics and
gradient distributions during training.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from utils.visualization.plot_utils import create_figure, save_figure, ensure_directory_exists

def plot_ghm_statistics(criterion, outputs1, outputs2, targets, save_dir, name):
    """Plot and save GHM statistics.
    
    Args:
        criterion: GHMRankingLoss instance
        outputs1 (torch.Tensor): Model outputs for first samples
        outputs2 (torch.Tensor): Model outputs for second samples
        targets (torch.Tensor): Target values
        save_dir (str): Directory to save plot files
        name (str): Base name for plot files
        
    Returns:
        dict: Statistics from the GHM calculation
    """
    from utils import ghm_utils  # Import here to avoid circular imports
    
    # Calculate GHM statistics
    stats = ghm_utils.calculate_ghm_statistics(
        criterion, outputs1, outputs2, targets,
        save_dir=save_dir,
        name=name
    )
    
    # Create additional visualization if needed
    plot_gradient_distribution(criterion, save_dir=save_dir, name=name)
    
    return stats

def plot_gradient_distribution(criterion, save_dir, name, figsize=(10, 8)):
    """Plot the gradient distribution from GHM loss.
    
    Args:
        criterion: GHMRankingLoss instance
        save_dir (str): Directory to save the plot
        name (str): Base name for the plot file
        figsize (tuple): Figure size in inches
        
    Returns:
        str: Path to the saved plot file
    """
    # Skip if criterion is not GHMRankingLoss or edges not initialized
    if not hasattr(criterion, 'edges') or not hasattr(criterion, 'distribution'):
        return None
    
    fig = create_figure(figsize)
    
    # Plot distribution histogram
    edges = criterion.edges.cpu().numpy()
    bins = len(edges) - 1
    
    if hasattr(criterion, 'distribution') and criterion.distribution is not None:
        dist = criterion.distribution.cpu().numpy()
        bin_widths = np.diff(edges)
        plt.bar(edges[:-1], dist / bin_widths, width=bin_widths, alpha=0.7)
    
    # Plot edges
    for edge in edges:
        plt.axvline(x=edge, color='r', linestyle='--', alpha=0.3)
    
    plt.xlabel('Gradient Magnitude')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    plt.title('Gradient Distribution with GHM Edges')
    
    # Save figure
    ensure_directory_exists(save_dir)
    save_path = os.path.join(save_dir, f"{name}_gradient_dist.png")
    save_figure(fig, save_path)
    
    return save_path 