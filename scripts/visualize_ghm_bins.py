#!/usr/bin/env python
"""
GHM Bin Visualization Script

This script visualizes the bin distribution and weights from GHM training,
helping to understand how the gradient harmonizing mechanism is working.

Usage:
    python scripts/visualize_ghm_bins.py --stats-dir <stats_dir> --output-dir <dir>
    
Example:
    python scripts/visualize_ghm_bins.py --stats-dir saved_models/stats/plastic_1000hz_ghm_20250418_161605 --output-dir results/ghm_viz
"""

import os
import argparse
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import re
from matplotlib.colors import LinearSegmentedColormap

# Add the root directory to the Python path to ensure imports work correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Visualize GHM bin distribution")
    
    parser.add_argument('--stats-dir', type=str, required=True,
                      help='Directory containing GHM statistics files')
    
    parser.add_argument('--output-dir', type=str, default='results/ghm_visualization',
                      help='Directory to save visualization results')
    
    parser.add_argument('--epochs', type=str, default='all',
                      help='Comma-separated epochs to analyze, or "all"')
    
    parser.add_argument('--alpha', type=float, default=0.75,
                      help='Alpha parameter used in GHM training (for weight calculation)')
    
    parser.add_argument('--create-animation', action='store_true',
                      help='Create animated visualization of bin changes')
    
    return parser.parse_args()

def get_stats_files(stats_dir):
    """Get GHM stats files sorted by epoch and batch."""
    files = glob.glob(os.path.join(stats_dir, "epoch*_batch*_*.npy"))
    
    # Sort files by epoch and batch
    def extract_epoch_batch(filename):
        match = re.search(r'epoch(\d+)_batch(\d+)', os.path.basename(filename))
        if match:
            return (int(match.group(1)), int(match.group(2)))
        return (0, 0)
    
    return sorted(files, key=extract_epoch_batch)

def load_ghm_stats(file_path):
    """Load GHM statistics from file."""
    try:
        stats = np.load(file_path, allow_pickle=True).item()
        return stats
    except Exception as e:
        print(f"Error loading stats from {file_path}: {e}")
        return None

def visualize_single_epoch(stats_file, output_path, alpha=0.75):
    """Visualize bin distribution for a single epoch."""
    stats = load_ghm_stats(stats_file)
    if not stats:
        return None
    
    bin_counts = stats['bin_counts']
    # Convert to numpy array if it's a list
    bin_counts = np.array(bin_counts)
    n_bins = len(bin_counts)
    
    # Calculate weights using GHM formula
    bin_weights = np.zeros_like(bin_counts, dtype=float)
    valid_indices = bin_counts > 0
    bin_weights[valid_indices] = 1.0 / (bin_counts[valid_indices] ** alpha)
    
    # Extract epoch and batch from filename
    match = re.search(r'epoch(\d+)_batch(\d+)', os.path.basename(stats_file))
    epoch = int(match.group(1)) if match else 0
    batch = int(match.group(2)) if match else 0
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot bin counts
    bars1 = ax1.bar(range(n_bins), bin_counts, color='skyblue', alpha=0.7)
    ax1.set_title(f'GHM Bin Counts - Epoch {epoch}, Batch {batch}')
    ax1.set_xlabel('Bin Index')
    ax1.set_ylabel('Sample Count')
    ax1.grid(alpha=0.3)
    
    # Add count labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}',
                    ha='center', va='bottom', rotation=0,
                    fontsize=8)
    
    # Plot bin weights
    bars2 = ax2.bar(range(n_bins), bin_weights, color='salmon', alpha=0.7)
    ax2.set_title(f'GHM Bin Weights (alpha={alpha}) - Epoch {epoch}, Batch {batch}')
    ax2.set_xlabel('Bin Index')
    ax2.set_ylabel('Weight')
    ax2.grid(alpha=0.3)
    
    # Add weight labels on bars
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}',
                    ha='center', va='bottom', rotation=0,
                    fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return output_path

def create_heatmap_visualization(stats_files, output_path, alpha=0.75):
    """Create a heatmap visualization of bin counts over epochs."""
    epoch_batch_data = []
    
    for file in stats_files:
        match = re.search(r'epoch(\d+)_batch(\d+)', os.path.basename(file))
        if match and match.group(2) == '0':  # Only first batch of each epoch
            epoch = int(match.group(1))
            stats = load_ghm_stats(file)
            if stats:
                bin_counts = np.array(stats['bin_counts'])  # Convert to numpy array
                epoch_batch_data.append((epoch, bin_counts))
    
    if not epoch_batch_data:
        print("No valid data found for heatmap visualization")
        return None
    
    # Sort by epoch
    epoch_batch_data.sort(key=lambda x: x[0])
    
    # Extract epochs and bin counts
    epochs = [x[0] for x in epoch_batch_data]
    bin_counts_matrix = np.array([x[1] for x in epoch_batch_data])
    
    # Create heatmap
    plt.figure(figsize=(14, 8))
    
    # Custom colormap from blue (low) to red (high)
    cmap = LinearSegmentedColormap.from_list('custom_heatmap', ['#ffffff', '#0571b0', '#92c5de', '#f7f7f7', '#f4a582', '#ca0020'])
    
    # Use log scale for better visualization since counts can vary widely
    plt.pcolormesh(np.arange(bin_counts_matrix.shape[1]), 
                 np.array(epochs), 
                 np.log1p(bin_counts_matrix),  # log(1+x) for better visualization
                 cmap=cmap,
                 shading='auto')
    
    plt.colorbar(label='Log(Count+1)')
    plt.title('GHM Bin Counts Evolution Over Epochs')
    plt.xlabel('Bin Index')
    plt.ylabel('Epoch')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    # Create heatmap for weights
    weights_matrix = np.zeros_like(bin_counts_matrix, dtype=float)
    valid_indices = bin_counts_matrix > 0
    weights_matrix[valid_indices] = 1.0 / (bin_counts_matrix[valid_indices] ** alpha)
    
    plt.figure(figsize=(14, 8))
    plt.pcolormesh(np.arange(weights_matrix.shape[1]), 
                 np.array(epochs), 
                 weights_matrix,
                 cmap='viridis',
                 shading='auto')
    
    plt.colorbar(label='Weight')
    plt.title(f'GHM Bin Weights Evolution Over Epochs (alpha={alpha})')
    plt.xlabel('Bin Index')
    plt.ylabel('Epoch')
    plt.tight_layout()
    
    weights_path = os.path.join(os.path.dirname(output_path), 'weights_heatmap.png')
    plt.savefig(weights_path, dpi=150)
    plt.close()
    
    return output_path

def create_bin_evolution_animation(stats_files, output_path, alpha=0.75):
    """Create an animated visualization of bin distribution changes."""
    # Filter to first batch of each epoch
    epoch_data = []
    for file in stats_files:
        match = re.search(r'epoch(\d+)_batch(\d+)', os.path.basename(file))
        if match and match.group(2) == '0':  # Only first batch of each epoch
            epoch = int(match.group(1))
            stats = load_ghm_stats(file)
            if stats:
                bin_counts = np.array(stats['bin_counts'])  # Convert to numpy array
                epoch_data.append((epoch, bin_counts))
    
    if not epoch_data:
        print("No valid data found for animation")
        return None
    
    # Sort by epoch
    epoch_data.sort(key=lambda x: x[0])
    
    # Setup the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Find max values for consistent y-axis
    max_count = max([np.max(counts) for _, counts in epoch_data])
    n_bins = len(epoch_data[0][1])
    
    # Create the initial empty plot
    bars1 = ax1.bar(range(n_bins), np.zeros(n_bins), color='skyblue', alpha=0.7)
    ax1.set_title('GHM Bin Counts')
    ax1.set_xlabel('Bin Index')
    ax1.set_ylabel('Sample Count')
    ax1.set_ylim(0, max_count * 1.1)
    ax1.grid(alpha=0.3)
    
    bars2 = ax2.bar(range(n_bins), np.zeros(n_bins), color='salmon', alpha=0.7)
    ax2.set_title(f'GHM Bin Weights (alpha={alpha})')
    ax2.set_xlabel('Bin Index')
    ax2.set_ylabel('Weight')
    ax2.set_ylim(0, 5)  # Set a reasonable y-limit for weights
    ax2.grid(alpha=0.3)
    
    # Text for epoch display
    epoch_text = ax1.text(0.98, 0.95, '', transform=ax1.transAxes, 
                        ha='right', va='top', fontsize=12)
    
    def update(frame):
        epoch, bin_counts = epoch_data[frame]
        
        # Update bin counts
        for i, bar in enumerate(bars1):
            bar.set_height(bin_counts[i])
        
        # Calculate and update bin weights
        bin_weights = np.zeros_like(bin_counts, dtype=float)
        valid_indices = bin_counts > 0
        bin_weights[valid_indices] = 1.0 / (bin_counts[valid_indices] ** alpha)
        
        for i, bar in enumerate(bars2):
            bar.set_height(bin_weights[i])
        
        # Update epoch text
        epoch_text.set_text(f'Epoch: {epoch}')
        
        return bars1, bars2, epoch_text
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(epoch_data), 
                                interval=300, blit=False)
    
    # Save animation
    ani.save(output_path, writer='pillow', fps=2, dpi=150)
    plt.close()
    
    return output_path

def main():
    """Main function for GHM bin visualization."""
    args = parse_arguments()
    
    stats_dir = args.stats_dir
    output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing GHM stats from: {stats_dir}")
    print(f"Saving visualizations to: {output_dir}")
    
    # Get all stats files
    all_files = get_stats_files(stats_dir)
    
    if not all_files:
        print(f"No GHM stats files found in {stats_dir}")
        return
    
    print(f"Found {len(all_files)} stats files")
    
    # Filter to specific epochs if requested
    if args.epochs != 'all':
        try:
            selected_epochs = [int(e) for e in args.epochs.split(',')]
            print(f"Selecting epochs: {selected_epochs}")
            filtered_files = []
            for file in all_files:
                match = re.search(r'epoch(\d+)', os.path.basename(file))
                if match and int(match.group(1)) in selected_epochs:
                    filtered_files.append(file)
            all_files = filtered_files
        except Exception as e:
            print(f"Error parsing epochs: {e}")
            print("Processing all epochs instead")
    
    # Process first batch of each epoch
    epoch_batch0_files = []
    for file in all_files:
        match = re.search(r'epoch(\d+)_batch(\d+)', os.path.basename(file))
        if match and match.group(2) == '0':  # Only first batch of each epoch
            epoch_batch0_files.append(file)
    
    # Visualize each selected epoch
    for file in epoch_batch0_files:
        match = re.search(r'epoch(\d+)', os.path.basename(file))
        if match:
            epoch = match.group(1)
            output_path = os.path.join(output_dir, f'epoch{epoch}_bins.png')
            print(f"Visualizing epoch {epoch}")
            visualize_single_epoch(file, output_path, alpha=args.alpha)
    
    # Create heatmap visualization
    print("Creating bin distribution heatmap")
    heatmap_path = os.path.join(output_dir, 'bins_heatmap.png')
    create_heatmap_visualization(all_files, heatmap_path, alpha=args.alpha)
    
    # Create animation if requested
    if args.create_animation:
        print("Creating bin evolution animation")
        animation_path = os.path.join(output_dir, 'bins_evolution.gif')
        create_bin_evolution_animation(all_files, animation_path, alpha=args.alpha)
    
    print(f"Visualization complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main() 