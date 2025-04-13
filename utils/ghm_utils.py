import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_gradient_distribution(ghm_loss, save_dir='./ghm_plots', name='ghm_distribution'):
    """
    Plot the gradient distribution and weights for GHM loss.
    
    Args:
        ghm_loss: The GHM loss object
        save_dir: Directory to save the plot
        name: Base name for the plot file
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get the current weights and move to CPU for plotting
    weights = ghm_loss.weights.cpu().numpy()
    
    # Plot the gradient density
    plt.figure(figsize=(10, 6))
    
    # Plot the weights for each bin
    bin_centers = (ghm_loss.edges.cpu()[:-1] + ghm_loss.edges.cpu()[1:]) / 2
    bin_centers = bin_centers.numpy()
    
    plt.bar(bin_centers, weights, width=1/ghm_loss.bins, alpha=0.7)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.xlabel('Gradient Magnitude')
    plt.ylabel('Weight')
    plt.title('GHM Gradient Distribution and Weights')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(save_dir, f'{name}.png'))
    plt.close()

def calculate_ghm_statistics(ghm_loss, outputs1, outputs2, targets, save_dir='./ghm_stats', name='ghm_stats'):
    """
    Calculate and save statistics related to GHM loss.
    
    Args:
        ghm_loss: The GHM loss object
        outputs1, outputs2: The model outputs for ranking pairs
        targets: The targets indicating which output should be higher
        save_dir: Directory to save the statistics
        name: Base name for the stats file
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get device
    device = outputs1.device
    
    # Compute loss gradients
    diff = outputs1 - outputs2
    expected_sign = 2 * targets - 1  # Convert 0/1 to -1/1
    g = torch.abs(torch.tanh(diff) - torch.tanh(ghm_loss.margin * expected_sign))
    
    # Get statistics
    g_cpu = g.detach().cpu().numpy()
    
    # Save statistics
    stats = {
        'mean_gradient': np.mean(g_cpu),
        'median_gradient': np.median(g_cpu),
        'min_gradient': np.min(g_cpu),
        'max_gradient': np.max(g_cpu),
        'std_gradient': np.std(g_cpu),
        'bin_counts': [],
        'bin_edges': ghm_loss.edges.cpu().numpy()
    }
    
    # Count samples in each bin
    edges = ghm_loss.edges.to(device)
    for i in range(ghm_loss.bins):
        inds = (g >= edges[i]) & (g < edges[i+1])
        num_in_bin = inds.sum().item()
        stats['bin_counts'].append(num_in_bin)
    
    # Save as numpy file
    np.save(os.path.join(save_dir, f'{name}.npy'), stats)
    
    return stats

def compare_losses(ghm_loss, regular_loss, outputs1, outputs2, targets):
    """
    Compare GHM loss with regular loss.
    
    Args:
        ghm_loss: The GHM loss object
        regular_loss: The regular loss (e.g., MarginRankingLoss)
        outputs1, outputs2: The model outputs for ranking pairs
        targets: The targets indicating which output should be higher
        
    Returns:
        Dictionary with loss comparisons
    """
    with torch.no_grad():
        # Compute GHM loss
        ghm_loss_val = ghm_loss(outputs1, outputs2, targets)
        
        # Compute regular loss
        reg_loss_val = regular_loss(outputs1, outputs2, targets)
        
        # Basic prediction statistics
        predictions = (outputs1 > outputs2)
        targets_bool = (targets > 0)
        correct = (predictions == targets_bool).float().mean()
        
        return {
            'ghm_loss': ghm_loss_val.item(),
            'regular_loss': reg_loss_val.item(),
            'accuracy': correct.item()
        } 