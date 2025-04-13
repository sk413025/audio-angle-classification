"""
GHM (Gradient Harmonizing Mechanism) Analyzer module.

This module provides tools for visualizing and analyzing GHM statistics
collected during model training with GHM loss.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import re
import torch
from typing import Union, List, Dict, Tuple, Optional, Any

from utils.visualization.plot_utils import create_figure, save_figure, ensure_directory_exists

def load_ghm_stats(stats_file: str) -> Dict[str, Any]:
    """Load GHM statistics from a .npy file.
    
    Args:
        stats_file: Path to the .npy file containing GHM statistics
        
    Returns:
        Dictionary containing the loaded statistics
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file is not a valid GHM statistics file
    """
    if not os.path.exists(stats_file):
        raise FileNotFoundError(f"GHM statistics file not found: {stats_file}")
    
    try:
        stats = np.load(stats_file, allow_pickle=True).item()
        
        # Validate that this is a GHM stats file
        required_keys = ['bin_counts', 'bin_edges', 'mean_gradient']
        if not all(key in stats for key in required_keys):
            raise ValueError(f"File {stats_file} is not a valid GHM statistics file")
            
        return stats
    except Exception as e:
        raise ValueError(f"Failed to load GHM statistics from {stats_file}: {str(e)}")

def find_ghm_stats_files(stats_dir: str, epoch: Optional[int] = None, 
                        batch: Optional[int] = None, 
                        pattern: Optional[str] = None) -> List[str]:
    """Find GHM statistics files in a directory.
    
    Args:
        stats_dir: Directory containing GHM statistics files
        epoch: Specific epoch to find (optional)
        batch: Specific batch index to find (optional)
        pattern: Custom glob pattern to use (optional)
        
    Returns:
        List of file paths matching the criteria
    """
    if not os.path.exists(stats_dir) or not os.path.isdir(stats_dir):
        raise FileNotFoundError(f"GHM statistics directory not found: {stats_dir}")
    
    # Build pattern based on parameters
    if pattern:
        search_pattern = os.path.join(stats_dir, pattern)
    elif epoch is not None and batch is not None:
        search_pattern = os.path.join(stats_dir, f"epoch{epoch}_batch{batch}_*.npy")
    elif epoch is not None:
        search_pattern = os.path.join(stats_dir, f"epoch{epoch}_batch*_*.npy")
    elif batch is not None:
        search_pattern = os.path.join(stats_dir, f"epoch*_batch{batch}_*.npy")
    else:
        search_pattern = os.path.join(stats_dir, "epoch*_batch*_*.npy")
    
    # Find matching files
    files = sorted(glob(search_pattern))
    return files

def parse_epoch_batch_from_filename(filename: str) -> Tuple[int, int]:
    """Extract epoch and batch information from a filename.
    
    Args:
        filename: GHM statistics filename
        
    Returns:
        Tuple of (epoch, batch) as integers
        
    Raises:
        ValueError: If epoch and batch cannot be extracted
    """
    basename = os.path.basename(filename)
    match = re.match(r'epoch(\d+)_batch(\d+)_', basename)
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        raise ValueError(f"Could not extract epoch and batch from filename: {basename}")

def visualize_ghm_stats(stats_or_file: Union[Dict[str, Any], str], 
                        output_path: Optional[str] = None, 
                        figsize: Tuple[int, int] = (16, 8),
                        show_plot: bool = False) -> str:
    """Visualize GHM statistics.
    
    Args:
        stats_or_file: GHM statistics dictionary or path to .npy file
        output_path: Path to save the visualization (optional)
        figsize: Figure size in inches (width, height)
        show_plot: Whether to display the plot
        
    Returns:
        Path to the saved visualization file (if output_path is provided)
    """
    # Load statistics if a file path is provided
    if isinstance(stats_or_file, str):
        stats = load_ghm_stats(stats_or_file)
        filename = os.path.basename(stats_or_file)
    else:
        stats = stats_or_file
        filename = "ghm_stats"
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Gradient distribution
    bin_counts = stats['bin_counts']
    bin_edges = stats['bin_edges']
    
    # Calculate bin centers for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    ax1.bar(range(len(bin_counts)), bin_counts, alpha=0.7)
    ax1.set_xlabel('Bin Index')
    ax1.set_ylabel('Sample Count')
    ax1.set_title('GHM Gradient Distribution by Bin')
    ax1.grid(True, alpha=0.3)
    
    # Add bin edge markers
    for i, edge in enumerate(bin_edges):
        ax1.axvline(x=i, color='r', linestyle='--', alpha=0.3)
    
    # Plot 2: Gradient statistics
    stat_names = ['mean_gradient', 'median_gradient', 'min_gradient', 'max_gradient', 'std_gradient']
    stat_labels = ['Mean', 'Median', 'Min', 'Max', 'Std']
    stat_values = [stats.get(name, 0) for name in stat_names]
    
    ax2.bar(stat_labels, stat_values, alpha=0.7)
    ax2.set_title('Gradient Statistics')
    ax2.grid(True, alpha=0.3)
    
    # Add file info to the plot
    plt.figtext(0.5, 0.01, filename, ha='center')
    
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        ensure_directory_exists(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=150)
        if not show_plot:
            plt.close(fig)
        return output_path
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return None

def compare_ghm_epochs(stats_dir: str, 
                       epochs: List[int], 
                       batch_idx: int = 0, 
                       output_path: Optional[str] = None,
                       figsize: Tuple[int, int] = (16, 12),
                       show_plot: bool = False) -> Optional[str]:
    """Compare GHM statistics across multiple epochs.
    
    Args:
        stats_dir: Directory containing GHM statistics files
        epochs: List of epochs to compare
        batch_idx: Batch index to use (0 for first batch, -1 for last batch)
        output_path: Path to save the comparison plot
        figsize: Figure size in inches
        show_plot: Whether to display the plot
        
    Returns:
        Path to the saved comparison plot (if output_path is provided)
    """
    epoch_data = []
    
    # Load data for each epoch
    for epoch in epochs:
        if batch_idx == -1:
            # Get the last batch
            pattern = os.path.join(stats_dir, f'epoch{epoch}_batch*_*.npy')
            files = sorted(glob(pattern))
            if not files:
                print(f"No files found for epoch {epoch}")
                continue
            file_path = files[-1]  # Take the last one
        else:
            # Get specific batch
            files = find_ghm_stats_files(stats_dir, epoch=epoch, batch=batch_idx)
            if not files:
                print(f"No files found for epoch {epoch}, batch {batch_idx}")
                continue
            file_path = files[0]
        
        # Load the stats
        try:
            stats = load_ghm_stats(file_path)
            epoch_data.append((epoch, stats))
            print(f"Loaded data for epoch {epoch}")
        except Exception as e:
            print(f"Error loading stats for epoch {epoch}: {str(e)}")
    
    if not epoch_data:
        print("No data loaded for comparison")
        return None
    
    # Create plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # Colors for different epochs
    colors = plt.cm.viridis(np.linspace(0, 1, len(epoch_data)))
    
    # 1. Compare bin distributions
    for i, (epoch, stats) in enumerate(epoch_data):
        bin_counts = stats['bin_counts']
        ax1.plot(range(len(bin_counts)), bin_counts, 
                marker='o', label=f'Epoch {epoch}', 
                color=colors[i], alpha=0.7)
    
    ax1.set_xlabel('Bin Index')
    ax1.set_ylabel('Sample Count')
    ax1.set_title('GHM Gradient Distribution by Bin')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Compare mean gradients
    epochs_list = [e for e, _ in epoch_data]
    mean_grads = [stats['mean_gradient'] for _, stats in epoch_data]
    
    ax2.plot(epochs_list, mean_grads, marker='o', linestyle='-', color='blue')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Gradient')
    ax2.set_title('Mean Gradient Across Epochs')
    ax2.grid(True, alpha=0.3)
    
    # 3. Compare gradient statistics
    width = 0.15
    stats_to_plot = ['mean_gradient', 'median_gradient', 'min_gradient', 'max_gradient', 'std_gradient']
    x = np.arange(len(stats_to_plot))
    
    for i, (epoch, stats) in enumerate(epoch_data):
        values = [stats.get(key, 0) for key in stats_to_plot]
        ax3.bar(x + i*width, values, width, label=f'Epoch {epoch}', alpha=0.7, color=colors[i])
    
    ax3.set_xlabel('Statistic')
    ax3.set_xticks(x + width * (len(epoch_data) - 1) / 2)
    ax3.set_xticklabels(['Mean', 'Median', 'Min', 'Max', 'Std'])
    ax3.set_title('Gradient Statistics Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Compare distribution heat map
    bin_data = np.array([stats['bin_counts'] for _, stats in epoch_data])
    im = ax4.imshow(bin_data, aspect='auto', cmap='viridis')
    ax4.set_xlabel('Bin Index')
    ax4.set_ylabel('Epoch')
    ax4.set_yticks(range(len(epochs_list)))
    ax4.set_yticklabels(epochs_list)
    ax4.set_title('Gradient Distribution Evolution')
    plt.colorbar(im, ax=ax4, label='Sample Count')
    
    plt.figtext(0.5, 0.01, f"Comparison of GHM Statistics (Batch {batch_idx})", ha='center', fontsize=12)
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        ensure_directory_exists(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=150)
        if not show_plot:
            plt.close(fig)
        return output_path
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return None

def analyze_ghm_training(stats_dir: str, 
                         output_dir: Optional[str] = None, 
                         selected_epochs: Optional[List[int]] = None,
                         max_epochs: int = 10) -> Dict[str, str]:
    """Analyze GHM statistics across an entire training run.
    
    This function performs comprehensive analysis:
    1. Individual epoch visualizations for selected epochs
    2. Comparison between beginning, middle, and end epochs
    3. Trend analysis for mean gradient and bin distributions
    
    Args:
        stats_dir: Directory containing GHM statistics files
        output_dir: Directory to save analysis outputs
        selected_epochs: Specific epochs to analyze (optional)
        max_epochs: Maximum number of epochs to include if selected_epochs is None
        
    Returns:
        Dictionary mapping analysis type to output file path
    """
    if not os.path.exists(stats_dir):
        raise FileNotFoundError(f"GHM statistics directory not found: {stats_dir}")
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(stats_dir, 'analysis')
    
    ensure_directory_exists(output_dir)
    
    # Find all unique epochs
    all_files = find_ghm_stats_files(stats_dir)
    all_epochs = set()
    
    for file in all_files:
        try:
            epoch, _ = parse_epoch_batch_from_filename(file)
            all_epochs.add(epoch)
        except ValueError:
            continue
    
    all_epochs = sorted(list(all_epochs))
    
    if not all_epochs:
        print(f"No epoch data found in {stats_dir}")
        return {}
    
    # Determine which epochs to analyze
    if selected_epochs:
        epochs_to_analyze = [e for e in selected_epochs if e in all_epochs]
    else:
        # If too many epochs, select a representative sample
        if len(all_epochs) > max_epochs:
            # Get start, end, and evenly spaced epochs in between
            step = max(1, len(all_epochs) // max_epochs)
            epochs_to_analyze = [all_epochs[0]]  # Always include first epoch
            
            # Add middle epochs
            for i in range(step, len(all_epochs) - 1, step):
                epochs_to_analyze.append(all_epochs[i])
            
            # Always include last epoch
            if all_epochs[-1] not in epochs_to_analyze:
                epochs_to_analyze.append(all_epochs[-1])
        else:
            epochs_to_analyze = all_epochs
    
    # Perform analyses
    outputs = {}
    
    # 1. Individual epoch visualizations
    print(f"Visualizing {len(epochs_to_analyze)} individual epochs...")
    for epoch in epochs_to_analyze:
        # Get first batch of the epoch
        files = find_ghm_stats_files(stats_dir, epoch=epoch, batch=0)
        if files:
            output_path = os.path.join(output_dir, f"epoch{epoch}_visualization.png")
            visualize_ghm_stats(files[0], output_path=output_path)
            outputs[f'epoch{epoch}_visualization'] = output_path
    
    # 2. Epoch comparisons
    # Compare first, middle, and last epoch if we have at least 3 epochs
    if len(epochs_to_analyze) >= 3:
        first_epoch = epochs_to_analyze[0]
        middle_idx = len(epochs_to_analyze) // 2
        middle_epoch = epochs_to_analyze[middle_idx]
        last_epoch = epochs_to_analyze[-1]
        
        comparison_epochs = [first_epoch, middle_epoch, last_epoch]
        
        # First batch comparison
        output_path = os.path.join(output_dir, "epoch_comparison_first_middle_last_batch0.png")
        compare_ghm_epochs(stats_dir, comparison_epochs, batch_idx=0, output_path=output_path)
        outputs['epoch_comparison_first_middle_last'] = output_path
        
        # Last batch comparison (if available)
        last_batch_files = find_ghm_stats_files(stats_dir, epoch=first_epoch)
        if len(last_batch_files) > 1:
            # Extract the batch number from the last file
            try:
                _, last_batch = parse_epoch_batch_from_filename(last_batch_files[-1])
                output_path = os.path.join(output_dir, f"epoch_comparison_first_middle_last_batch{last_batch}.png")
                compare_ghm_epochs(stats_dir, comparison_epochs, batch_idx=last_batch, output_path=output_path)
                outputs['epoch_comparison_first_middle_last_last_batch'] = output_path
            except ValueError:
                pass
    
    # 3. Comprehensive comparison of all selected epochs
    if len(epochs_to_analyze) > 1:
        output_path = os.path.join(output_dir, "all_epochs_comparison.png")
        compare_ghm_epochs(stats_dir, epochs_to_analyze, batch_idx=0, output_path=output_path)
        outputs['all_epochs_comparison'] = output_path
    
    print(f"Analysis complete. Generated {len(outputs)} visualizations in {output_dir}")
    return outputs

def calculate_ghm_statistics_from_criterion(criterion, outputs1, outputs2, targets):
    """Calculate GHM statistics directly from a criterion and model outputs.
    
    This is a helper function to generate GHM statistics for visualization
    without saving to a file first.
    
    Args:
        criterion: GHMRankingLoss instance
        outputs1: Model outputs for first samples
        outputs2: Model outputs for second samples
        targets: Target values
        
    Returns:
        Dictionary of GHM statistics
    """
    from utils import ghm_utils  # Import here to avoid circular imports
    
    # Use the existing function, but don't save to file
    stats = ghm_utils.calculate_ghm_statistics(
        criterion, outputs1, outputs2, targets,
        save_dir=None,
        name=None
    )
    
    return stats

def visualize_ghm_live(criterion, outputs1, outputs2, targets, 
                     output_path=None, show_plot=True):
    """Visualize GHM statistics directly from a criterion during training.
    
    Args:
        criterion: GHMRankingLoss instance
        outputs1: Model outputs for first samples
        outputs2: Model outputs for second samples
        targets: Target values
        output_path: Path to save the visualization
        show_plot: Whether to display the plot
        
    Returns:
        Path to the saved visualization if output_path is provided
    """
    # Calculate statistics
    stats = calculate_ghm_statistics_from_criterion(criterion, outputs1, outputs2, targets)
    
    # Visualize
    return visualize_ghm_stats(stats, output_path=output_path, show_plot=show_plot)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GHM Analysis Tools")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Visualize single file
    viz_parser = subparsers.add_parser("visualize", help="Visualize a single GHM statistics file")
    viz_parser.add_argument("--file", type=str, required=True, help="Path to GHM statistics file")
    viz_parser.add_argument("--output", type=str, help="Output path for visualization")
    viz_parser.add_argument("--show", action="store_true", help="Show the plot")
    
    # Compare epochs
    compare_parser = subparsers.add_parser("compare", help="Compare GHM statistics across epochs")
    compare_parser.add_argument("--dir", type=str, required=True, help="Directory containing GHM statistics files")
    compare_parser.add_argument("--epochs", type=int, nargs="+", required=True, help="Epochs to compare")
    compare_parser.add_argument("--batch", type=int, default=0, help="Batch index to use")
    compare_parser.add_argument("--output", type=str, help="Output path for comparison plot")
    compare_parser.add_argument("--show", action="store_true", help="Show the plot")
    
    # Analyze training
    analyze_parser = subparsers.add_parser("analyze", help="Analyze GHM statistics across training")
    analyze_parser.add_argument("--dir", type=str, required=True, help="Directory containing GHM statistics files")
    analyze_parser.add_argument("--output", type=str, help="Output directory for analysis")
    analyze_parser.add_argument("--epochs", type=int, nargs="+", help="Specific epochs to analyze")
    analyze_parser.add_argument("--max-epochs", type=int, default=10, help="Maximum number of epochs to include in analysis")
    
    args = parser.parse_args()
    
    if args.command == "visualize":
        visualize_ghm_stats(args.file, output_path=args.output, show_plot=args.show)
    
    elif args.command == "compare":
        compare_ghm_epochs(args.dir, args.epochs, batch_idx=args.batch, 
                          output_path=args.output, show_plot=args.show)
    
    elif args.command == "analyze":
        analyze_ghm_training(args.dir, output_dir=args.output, 
                            selected_epochs=args.epochs, max_epochs=args.max_epochs)
    
    else:
        parser.print_help() 