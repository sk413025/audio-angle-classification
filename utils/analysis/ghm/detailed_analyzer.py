"""
GHM Detailed Analyzer Module

This module provides detailed analysis of GHM training statistics:
- Bin evolution visualization
- Gradient statistics analysis
- Parameter comparison tools
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import re
import torch

from utils.visualization import (
    visualize_ghm_stats,
    compare_ghm_epochs,
    analyze_ghm_training
)

def get_stats_files(stats_dir):
    """Get all GHM stats files in a directory, sorted by epoch and batch."""
    files = glob(os.path.join(stats_dir, "epoch*_batch*_*.npy"))
    
    # Sort files by epoch and batch
    def extract_epoch_batch(filename):
        match = re.search(r'epoch(\d+)_batch(\d+)', os.path.basename(filename))
        if match:
            return (int(match.group(1)), int(match.group(2)))
        return (0, 0)
    
    return sorted(files, key=extract_epoch_batch)

def load_ghm_stats(file_path):
    """Load GHM statistics from a numpy file."""
    try:
        stats = np.load(file_path, allow_pickle=True).item()
        return stats
    except Exception as e:
        print(f"Error loading stats from {file_path}: {e}")
        return None

def visualize_bin_evolution(stats_dir, output_dir, selected_epochs=None):
    """Visualize how bins evolve over the course of training."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all stats files
    all_files = get_stats_files(stats_dir)
    
    # Group files by epoch
    epoch_files = {}
    for file in all_files:
        match = re.search(r'epoch(\d+)', os.path.basename(file))
        if match:
            epoch = int(match.group(1))
            if epoch not in epoch_files:
                epoch_files[epoch] = []
            epoch_files[epoch].append(file)
    
    # Get list of epochs
    epochs = sorted(epoch_files.keys())
    
    if selected_epochs:
        # Filter to only selected epochs
        epochs = [e for e in epochs if e in selected_epochs]
    elif len(epochs) > 10:
        # If too many epochs, select a representative subset
        step = max(len(epochs) // 10, 1)
        epochs = epochs[::step]
        # Always include first and last epoch
        if epochs[0] != 1:
            epochs = [1] + epochs
        if epochs[-1] != max(epoch_files.keys()):
            epochs.append(max(epoch_files.keys()))
    
    # For each epoch, get the first batch stats
    epoch_stats = []
    for epoch in epochs:
        if epoch_files[epoch]:
            stats = load_ghm_stats(epoch_files[epoch][0])
            epoch_stats.append((epoch, stats))
    
    # Plot bin counts evolution
    plt.figure(figsize=(15, 8))
    
    # Create a colormap from blue to red
    cmap = plt.get_cmap('coolwarm')
    colors = [cmap(i) for i in np.linspace(0, 1, len(epoch_stats))]
    
    for i, (epoch, stats) in enumerate(epoch_stats):
        bin_counts = stats['bin_counts']
        plt.plot(range(len(bin_counts)), bin_counts, 
                 marker='o', 
                 linestyle='-', 
                 linewidth=2,
                 label=f'Epoch {epoch}',
                 color=colors[i],
                 alpha=0.7)
    
    plt.title('Evolution of GHM Bin Counts During Training')
    plt.xlabel('Bin Index')
    plt.ylabel('Sample Count')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bin_counts_evolution.png'), dpi=150)
    plt.close()
    
    # Plot bin weights evolution
    plt.figure(figsize=(15, 8))
    
    for i, (epoch, stats) in enumerate(epoch_stats):
        bin_counts = np.array(stats['bin_counts'])
        # Calculate weights using the GHM formula
        bin_weights = np.power(np.maximum(bin_counts, 1), -0.75)  # Using default alpha=0.75
        
        plt.plot(range(len(bin_weights)), bin_weights, 
                 marker='o', 
                 linestyle='-', 
                 linewidth=2,
                 label=f'Epoch {epoch}',
                 color=colors[i],
                 alpha=0.7)
    
    plt.title('Evolution of GHM Bin Weights During Training')
    plt.xlabel('Bin Index')
    plt.ylabel('Weight')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bin_weights_evolution.png'), dpi=150)
    plt.close()
    
    # Return the paths to the generated plots
    return {
        'bin_counts_evolution': os.path.join(output_dir, 'bin_counts_evolution.png'),
        'bin_weights_evolution': os.path.join(output_dir, 'bin_weights_evolution.png')
    }

def analyze_gradient_statistics(stats_dir, output_dir):
    """Analyze gradient statistics evolution during training."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all stats files
    all_files = get_stats_files(stats_dir)
    
    # Collect statistics from each file
    epoch_stats = []
    for file in all_files:
        match = re.search(r'epoch(\d+)_batch(\d+)', os.path.basename(file))
        if match and match.group(2) == '0':  # Only use first batch of each epoch
            epoch = int(match.group(1))
            stats = load_ghm_stats(file)
            
            # Extract gradient statistics
            epoch_stats.append({
                'epoch': epoch,
                'mean': stats.get('mean_gradient', 0),
                'median': stats.get('median_gradient', 0),
                'min': stats.get('min_gradient', 0),
                'max': stats.get('max_gradient', 0),
                'std': stats.get('std_gradient', 0)
            })
    
    # Sort by epoch
    epoch_stats.sort(key=lambda x: x['epoch'])
    
    # Extract data for plotting
    epochs = [stat['epoch'] for stat in epoch_stats]
    mean_vals = [stat['mean'] for stat in epoch_stats]
    median_vals = [stat['median'] for stat in epoch_stats]
    min_vals = [stat['min'] for stat in epoch_stats]
    max_vals = [stat['max'] for stat in epoch_stats]
    std_vals = [stat['std'] for stat in epoch_stats]
    
    # Plot gradient statistics evolution
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(epochs, mean_vals, 'o-', label='Mean', linewidth=2)
    plt.plot(epochs, median_vals, 's-', label='Median', linewidth=2)
    plt.fill_between(epochs, 
                     np.array(mean_vals) - np.array(std_vals),
                     np.array(mean_vals) + np.array(std_vals),
                     alpha=0.2, label='Mean ± Std')
    plt.title('Gradient Statistics Evolution (Mean and Median)')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(epochs, min_vals, 'v-', label='Min', linewidth=2)
    plt.plot(epochs, max_vals, '^-', label='Max', linewidth=2)
    plt.title('Gradient Statistics Evolution (Min and Max)')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradient_statistics_evolution.png'), dpi=150)
    plt.close()
    
    # Plot distribution of gradients for selected epochs
    if len(epochs) > 3:
        # Select first, middle, and last epochs
        selected_epochs = [epochs[0], epochs[len(epochs)//2], epochs[-1]]
        
        plt.figure(figsize=(15, 5))
        for i, epoch in enumerate(selected_epochs):
            plt.subplot(1, 3, i+1)
            
            # Find stats for this epoch
            stat = next((s for s in epoch_stats if s['epoch'] == epoch), None)
            if stat:
                # Plot normal distribution based on mean and std
                x = np.linspace(stat['mean'] - 3*stat['std'], 
                                stat['mean'] + 3*stat['std'], 100)
                y = np.exp(-0.5 * ((x - stat['mean']) / stat['std'])**2) / (stat['std'] * np.sqrt(2*np.pi))
                
                plt.plot(x, y, linewidth=2)
                plt.axvline(stat['mean'], color='r', linestyle='--', label='Mean')
                plt.axvline(stat['median'], color='g', linestyle='--', label='Median')
                plt.title(f'Epoch {epoch} Gradient Distribution')
                plt.xlabel('Gradient Value')
                plt.ylabel('Density')
                plt.legend()
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gradient_distributions.png'), dpi=150)
        plt.close()
    
    return {
        'gradient_statistics_evolution': os.path.join(output_dir, 'gradient_statistics_evolution.png'),
        'gradient_distributions': os.path.join(output_dir, 'gradient_distributions.png') if len(epochs) > 3 else None
    }

def compare_different_ghm_params(base_dir, output_dir, params_config):
    """Compare the effect of different GHM parameters on training."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store results for each parameter configuration
    param_results = {}
    
    # Process each parameter configuration
    for param_name, param_pattern in params_config.items():
        # Find the directory matching the pattern
        param_dirs = glob(os.path.join(base_dir, param_pattern))
        
        if not param_dirs:
            print(f"No directories found for pattern: {param_pattern}")
            continue
            
        # Use the latest directory if multiple matches
        param_dir = sorted(param_dirs)[-1]
        
        # Check for GHM stats directory
        stats_dir = os.path.join(param_dir, 'ghm_stats')
        if not os.path.exists(stats_dir):
            print(f"No GHM stats directory found in {param_dir}")
            continue
            
        # Get all stats files
        all_files = get_stats_files(stats_dir)
        
        if not all_files:
            print(f"No stats files found in {stats_dir}")
            continue
            
        # Group by epoch (using first batch of each epoch)
        epoch_files = {}
        for file in all_files:
            match = re.search(r'epoch(\d+)_batch(\d+)', os.path.basename(file))
            if match and match.group(2) == '0':
                epoch_files[int(match.group(1))] = file
        
        # Get selected epochs (first, middle, last)
        epochs = sorted(epoch_files.keys())
        if not epochs:
            continue
            
        first_epoch = min(epochs)
        last_epoch = max(epochs)
        mid_epoch = epochs[len(epochs)//2]
        
        selected_epochs = [first_epoch, mid_epoch, last_epoch]
        epoch_stats = []
        
        for epoch in selected_epochs:
            if epoch in epoch_files:
                stats = load_ghm_stats(epoch_files[epoch])
                if stats:
                    epoch_stats.append((epoch, stats))
        
        if not epoch_stats:
            continue
            
        # Create output directory for this parameter
        param_output_dir = os.path.join(output_dir, param_name)
        os.makedirs(param_output_dir, exist_ok=True)
        
        # Visualize bin counts for this parameter
        plt.figure(figsize=(15, 8))
        
        for epoch, stats in epoch_stats:
            bin_counts = stats['bin_counts']
            plt.plot(range(len(bin_counts)), bin_counts, 
                    marker='o', 
                    linestyle='-', 
                    linewidth=2,
                    label=f'Epoch {epoch}')
        
        plt.title(f'{param_name}: Evolution of GHM Bin Counts')
        plt.xlabel('Bin Index')
        plt.ylabel('Sample Count')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(param_output_dir, 'bin_counts.png'), dpi=150)
        plt.close()
        
        # Store results for comparison
        param_results[param_name] = {
            'stats_dir': stats_dir,
            'epoch_stats': epoch_stats,
            'output_dir': param_output_dir,
            'bin_counts_plot': os.path.join(param_output_dir, 'bin_counts.png')
        }
    
    # Create comparison plots between different parameters
    if len(param_results) > 1:
        # Compare bin counts for the last epoch of each parameter
        plt.figure(figsize=(15, 8))
        
        for param_name, result in param_results.items():
            # Get the last epoch stats
            last_epoch_stat = result['epoch_stats'][-1]
            bin_counts = last_epoch_stat[1]['bin_counts']
            
            plt.plot(range(len(bin_counts)), bin_counts, 
                    marker='o', 
                    linestyle='-', 
                    linewidth=2,
                    label=f'{param_name} (Epoch {last_epoch_stat[0]})')
        
        plt.title('Comparison of GHM Bin Counts Across Parameters')
        plt.xlabel('Bin Index')
        plt.ylabel('Sample Count')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'bin_counts_comparison.png'), dpi=150)
        plt.close()
    
    return param_results

def analyze_ghm_details(base_dir, output_dir, frequency='1000hz', param_type=None, param_value=None):
    """Main function to perform detailed GHM analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine the pattern to match based on parameters
    if param_type and param_value:
        pattern = f"*ghm_{frequency}_{param_type}{param_value}*"
    else:
        pattern = f"*ghm_{frequency}*"
    
    # Find training directories
    dirs = glob(os.path.join(base_dir, pattern))
    if not dirs:
        print(f"No directories found matching pattern: {pattern}")
        return {}
    
    # Use the latest directory
    training_dir = sorted(dirs)[-1]
    
    # Check for GHM stats directory
    stats_dir = os.path.join(training_dir, 'ghm_stats')
    if not os.path.exists(stats_dir):
        print(f"No GHM stats directory found in {training_dir}")
        return {}
    
    # Create subdirectories for different analyses
    bin_evolution_dir = os.path.join(output_dir, 'bin_evolution')
    gradient_stats_dir = os.path.join(output_dir, 'gradient_stats')
    
    # Run different analyses
    bin_results = visualize_bin_evolution(stats_dir, bin_evolution_dir)
    gradient_results = analyze_gradient_statistics(stats_dir, gradient_stats_dir)
    
    # If parameter comparison is requested, set up the configurations
    if param_type:
        param_comparison_dir = os.path.join(output_dir, 'param_comparison')
        
        if param_type == 'alpha':
            params_config = {
                'alpha_0.5': f"*ghm_{frequency}_alpha0.5*",
                'alpha_0.75': f"*ghm_{frequency}*",  # Default
                'alpha_0.9': f"*ghm_{frequency}_alpha0.9*"
            }
        elif param_type == 'bins':
            params_config = {
                'bins_5': f"*ghm_{frequency}_bins5*",
                'bins_10': f"*ghm_{frequency}*",  # Default
                'bins_20': f"*ghm_{frequency}_bins20*"
            }
        else:
            params_config = {}
        
        if params_config:
            param_results = compare_different_ghm_params(base_dir, param_comparison_dir, params_config)
            
            # Add to overall results
            results = {
                'bin_evolution': bin_results,
                'gradient_stats': gradient_results,
                'param_comparison': param_results
            }
        else:
            results = {
                'bin_evolution': bin_results,
                'gradient_stats': gradient_results
            }
    else:
        results = {
            'bin_evolution': bin_results,
            'gradient_stats': gradient_results
        }
    
    return results