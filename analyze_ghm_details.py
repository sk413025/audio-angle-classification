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
    
    # Plot standard deviation evolution
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, std_vals, 'o-', linewidth=2, color='purple')
    plt.title('Gradient Standard Deviation Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Standard Deviation')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradient_std_evolution.png'), dpi=150)
    plt.close()
    
    return {
        'gradient_statistics_evolution': os.path.join(output_dir, 'gradient_statistics_evolution.png'),
        'gradient_std_evolution': os.path.join(output_dir, 'gradient_std_evolution.png')
    }

def compare_different_ghm_params(base_dir, output_dir, params_config):
    """Compare GHM statistics across different parameter configurations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect stats from the last epoch of each configuration
    last_epoch_stats = {}
    
    for config_name, stats_dir in params_config.items():
        # Find the last epoch stats file
        files = get_stats_files(stats_dir)
        if not files:
            print(f"No stats files found for {config_name}")
            continue
        
        # Group by epoch and get the last one
        epoch_files = {}
        for file in files:
            match = re.search(r'epoch(\d+)', os.path.basename(file))
            if match:
                epoch = int(match.group(1))
                if epoch not in epoch_files:
                    epoch_files[epoch] = []
                epoch_files[epoch].append(file)
        
        if not epoch_files:
            print(f"No valid epoch files found for {config_name}")
            continue
        
        last_epoch = max(epoch_files.keys())
        last_file = epoch_files[last_epoch][0]  # Take the first batch of the last epoch
        
        # Load stats
        stats = load_ghm_stats(last_file)
        last_epoch_stats[config_name] = {
            'epoch': last_epoch,
            'stats': stats
        }
    
    # Compare bin counts
    plt.figure(figsize=(15, 8))
    
    for config_name, data in last_epoch_stats.items():
        bin_counts = data['stats']['bin_counts']
        plt.plot(range(len(bin_counts)), bin_counts, 
                 marker='o', 
                 linestyle='-', 
                 linewidth=2,
                 label=f'{config_name} (Epoch {data["epoch"]})')
    
    plt.title('Comparison of GHM Bin Counts with Different Parameters')
    plt.xlabel('Bin Index')
    plt.ylabel('Sample Count')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bin_counts_comparison.png'), dpi=150)
    plt.close()
    
    # Compare gradient statistics
    stats_data = {
        'mean': [],
        'median': [],
        'min': [],
        'max': [],
        'std': []
    }
    
    config_names = []
    
    for config_name, data in last_epoch_stats.items():
        stats = data['stats']
        config_names.append(config_name)
        stats_data['mean'].append(stats.get('mean_gradient', 0))
        stats_data['median'].append(stats.get('median_gradient', 0))
        stats_data['min'].append(stats.get('min_gradient', 0))
        stats_data['max'].append(stats.get('max_gradient', 0))
        stats_data['std'].append(stats.get('std_gradient', 0))
    
    # Plot gradient statistics comparison
    plt.figure(figsize=(15, 8))
    
    x = np.arange(len(config_names))
    width = 0.15
    
    plt.bar(x - 2*width, stats_data['mean'], width, label='Mean')
    plt.bar(x - width, stats_data['median'], width, label='Median')
    plt.bar(x, stats_data['min'], width, label='Min')
    plt.bar(x + width, stats_data['max'], width, label='Max')
    plt.bar(x + 2*width, stats_data['std'], width, label='Std')
    
    plt.title('Comparison of Gradient Statistics with Different Parameters')
    plt.xlabel('Configuration')
    plt.ylabel('Value')
    plt.xticks(x, config_names, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradient_statistics_comparison.png'), dpi=150)
    plt.close()
    
    return {
        'bin_counts_comparison': os.path.join(output_dir, 'bin_counts_comparison.png'),
        'gradient_statistics_comparison': os.path.join(output_dir, 'gradient_statistics_comparison.png')
    }

def main():
    """Main function to run GHM detailed analysis."""
    # Set up directories
    base_dir = "/Users/sbplab/Hank/angle_classification_deg6/saved_models"
    output_base_dir = "ghm_analysis_results/detailed_analysis"
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Find stats directories for each configuration
    stats_dirs = {}
    for freq in ['500hz', '1000hz', '3000hz']:
        pattern = os.path.join(base_dir, f"stats/plastic_{freq}_ghm_*")
        dirs = sorted(glob(pattern))
        if dirs:
            stats_dirs[freq] = dirs[-1]  # Use the most recent one
    
    # Find parameter variation stats directories
    param_stats_dirs = {}
    
    # Bins variation
    for bins in [5, 10, 20]:
        pattern = os.path.join(base_dir, f"stats/plastic_1000hz_ghm_*_bins{bins}_*")
        dirs = sorted(glob(pattern))
        if dirs:
            param_stats_dirs[f'bins{bins}'] = dirs[-1]
    
    # Alpha variation
    for alpha in [0.5, 0.75, 0.9]:
        alpha_str = str(alpha).replace('.', '')
        pattern = os.path.join(base_dir, f"stats/plastic_1000hz_ghm_*_alpha{alpha_str}_*")
        dirs = sorted(glob(pattern))
        if dirs:
            param_stats_dirs[f'alpha{alpha}'] = dirs[-1]
    
    # Analyze each frequency
    for freq, stats_dir in stats_dirs.items():
        if not os.path.exists(stats_dir):
            print(f"Stats directory not found: {stats_dir}")
            continue
            
        print(f"Analyzing frequency: {freq}")
        freq_output_dir = os.path.join(output_base_dir, freq)
        os.makedirs(freq_output_dir, exist_ok=True)
        
        # Visualize bin evolution
        bin_evolution_plots = visualize_bin_evolution(stats_dir, os.path.join(freq_output_dir, 'bin_evolution'))
        print(f"Bin evolution plots generated: {bin_evolution_plots}")
        
        # Analyze gradient statistics
        gradient_stats_plots = analyze_gradient_statistics(stats_dir, os.path.join(freq_output_dir, 'gradient_stats'))
        print(f"Gradient statistics plots generated: {gradient_stats_plots}")
    
    # Compare parameter variations
    if param_stats_dirs:
        print("Comparing different GHM parameters")
        params_output_dir = os.path.join(output_base_dir, 'parameter_comparison')
        os.makedirs(params_output_dir, exist_ok=True)
        
        # Compare bin variations
        bin_param_dirs = {k: v for k, v in param_stats_dirs.items() if k.startswith('bins')}
        if len(bin_param_dirs) > 1:
            bin_comparison_plots = compare_different_ghm_params(
                base_dir, 
                os.path.join(params_output_dir, 'bins_comparison'),
                bin_param_dirs
            )
            print(f"Bin parameter comparison plots generated: {bin_comparison_plots}")
        
        # Compare alpha variations
        alpha_param_dirs = {k: v for k, v in param_stats_dirs.items() if k.startswith('alpha')}
        if len(alpha_param_dirs) > 1:
            alpha_comparison_plots = compare_different_ghm_params(
                base_dir, 
                os.path.join(params_output_dir, 'alpha_comparison'),
                alpha_param_dirs
            )
            print(f"Alpha parameter comparison plots generated: {alpha_comparison_plots}")
    
    print(f"Detailed GHM analysis complete. Results saved to {output_base_dir}")

if __name__ == "__main__":
    main() 