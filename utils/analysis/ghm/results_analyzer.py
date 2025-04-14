"""
GHM Results Analyzer Module

This module analyzes GHM training results, providing functionality to:
- Find training directories
- Load training history
- Analyze GHM stats
- Plot comparison results
- Create parameter impact visualizations
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from glob import glob
import re
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib.animation import FuncAnimation

from utils.visualization import (
    visualize_ghm_stats,
    compare_ghm_epochs,
    analyze_ghm_training
)

def find_latest_training_dirs(base_dir, pattern):
    """Find the latest training directories matching a pattern."""
    dirs = glob(os.path.join(base_dir, pattern))
    # Sort by timestamp in directory name
    dirs = sorted(dirs, key=lambda x: re.search(r'(\d{8}_\d{6})', x).group(1) if re.search(r'(\d{8}_\d{6})', x) else '')
    return dirs[-1] if dirs else None

def load_training_history(history_path):
    """Load training history from pickle file."""
    with open(history_path, 'rb') as f:
        return pickle.load(f)

def plot_training_comparison(histories, labels, output_path):
    """Plot training history comparison between different models."""
    plt.figure(figsize=(16, 12))
    
    # Plot training loss
    plt.subplot(2, 2, 1)
    for history, label in zip(histories, labels):
        plt.plot(history['epoch'], history['train_loss_main'], label=f'{label} (Train)')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Plot validation loss
    plt.subplot(2, 2, 2)
    for history, label in zip(histories, labels):
        plt.plot(history['epoch'], history['val_loss_main'], label=f'{label} (Val)')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Plot training accuracy
    plt.subplot(2, 2, 3)
    for history, label in zip(histories, labels):
        plt.plot(history['epoch'], history['train_accuracy'], label=f'{label} (Train)')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Plot validation accuracy
    plt.subplot(2, 2, 4)
    for history, label in zip(histories, labels):
        plt.plot(history['epoch'], history['val_accuracy'], label=f'{label} (Val)')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return output_path

def analyze_single_run(stats_dir, plots_dir, output_dir):
    """Analyze a single GHM training run."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Find representative epoch stats files
    all_files = glob(os.path.join(stats_dir, "epoch*_batch0_*.npy"))
    epoch_nums = sorted([int(re.search(r'epoch(\d+)_', os.path.basename(f)).group(1)) for f in all_files])
    
    if not epoch_nums:
        print(f"No stats files found in {stats_dir}")
        return {}
        
    early_epoch = max(1, min(epoch_nums))
    mid_epoch = max(1, len(epoch_nums) // 2)
    late_epoch = max(epoch_nums)
    
    early_file = glob(os.path.join(stats_dir, f"epoch{early_epoch}_batch0_*.npy"))[0]
    mid_file = glob(os.path.join(stats_dir, f"epoch{mid_epoch}_batch0_*.npy"))[0]
    late_file = glob(os.path.join(stats_dir, f"epoch{late_epoch}_batch0_*.npy"))[0]
    
    # 2. Visualize individual epoch stats
    visualize_ghm_stats(early_file, os.path.join(output_dir, "early_epoch_ghm.png"))
    visualize_ghm_stats(mid_file, os.path.join(output_dir, "mid_epoch_ghm.png"))
    visualize_ghm_stats(late_file, os.path.join(output_dir, "late_epoch_ghm.png"))
    
    # 3. Compare epochs
    compare_ghm_epochs(
        stats_dir,
        epochs=[early_epoch, mid_epoch, late_epoch],
        output_path=os.path.join(output_dir, "epoch_comparison.png")
    )
    
    # 4. Analyze entire training process
    full_analysis_dir = os.path.join(output_dir, "full_training_analysis")
    os.makedirs(full_analysis_dir, exist_ok=True)
    analyze_ghm_training(stats_dir, output_dir=full_analysis_dir)
    
    return {
        'early_epoch_vis': os.path.join(output_dir, "early_epoch_ghm.png"),
        'mid_epoch_vis': os.path.join(output_dir, "mid_epoch_ghm.png"),
        'late_epoch_vis': os.path.join(output_dir, "late_epoch_ghm.png"),
        'epoch_comparison': os.path.join(output_dir, "epoch_comparison.png"),
        'full_analysis_dir': full_analysis_dir
    }

def plot_confusion_matrix(model_path, data_loader, output_path, class_names=None):
    """Generate and plot a confusion matrix for a trained model."""
    # Load the model
    model = torch.load(model_path)
    model.eval()
    
    # Determine device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    
    # Collect predictions and true labels
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for data1, data2, targets, label1, label2 in data_loader:
            data1 = data1.to(device)
            data2 = data2.to(device)
            
            # Get model outputs
            output1 = model(data1)
            output2 = model(data2)
            
            # Determine if model predicts correctly (output1 > output2 when target=1, output1 < output2 when target=-1)
            predictions = (output1 > output2).cpu().numpy()
            actual = (targets > 0).cpu().numpy()
            
            y_true.extend(actual)
            y_pred.extend(predictions)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    if class_names:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks + 0.5, class_names)
        plt.yticks(tick_marks + 0.5, class_names)
    
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return output_path

def create_parameter_impact_plot(results, output_path):
    """Create a bar chart showing the impact of different GHM parameters."""
    params = list(results.keys())
    accuracies = [results[param]['best_val_accuracy'] for param in params]
    
    plt.figure(figsize=(14, 8))
    bars = plt.bar(params, accuracies, alpha=0.7)
    
    # Add value labels on top of bars
    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{acc:.2f}%',
            ha='center',
            fontsize=10
        )
    
    plt.title('Impact of GHM Parameters on Validation Accuracy')
    plt.xlabel('Parameter Configuration')
    plt.ylabel('Best Validation Accuracy (%)')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return output_path

def create_frequency_comparison_plot(results, output_path):
    """Create a comparison plot for different frequency configurations."""
    frequencies = list(results.keys())
    accuracies = [results[freq]['best_val_accuracy'] for freq in frequencies]
    
    plt.figure(figsize=(14, 8))
    bars = plt.bar(frequencies, accuracies, alpha=0.7)
    
    # Add value labels on top of bars
    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{acc:.2f}%',
            ha='center',
            fontsize=10
        )
    
    plt.title('Comparison of Model Performance Across Frequencies')
    plt.xlabel('Frequency')
    plt.ylabel('Best Validation Accuracy (%)')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return output_path

def analyze_ghm_results(base_dir, output_dir):
    """Main function to analyze GHM training results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Find training directories for different frequencies and parameters
    frequencies = ['500hz', '1000hz', '3000hz']
    results = {}
    
    for freq in frequencies:
        freq_results = {}
        
        # Standard training
        standard_dir = find_latest_training_dirs(base_dir, f"*standard_{freq}*")
        if standard_dir:
            standard_history_path = os.path.join(standard_dir, 'training_history.pkl')
            if os.path.exists(standard_history_path):
                standard_history = load_training_history(standard_history_path)
                freq_results['standard'] = {
                    'history': standard_history,
                    'best_val_accuracy': max(standard_history['val_accuracy'])
                }
        
        # Default GHM
        ghm_dir = find_latest_training_dirs(base_dir, f"*ghm_{freq}_*")
        if ghm_dir:
            ghm_history_path = os.path.join(ghm_dir, 'training_history.pkl')
            ghm_stats_dir = os.path.join(ghm_dir, 'ghm_stats')
            ghm_plots_dir = os.path.join(ghm_dir, 'plots')
            
            if os.path.exists(ghm_history_path):
                ghm_history = load_training_history(ghm_history_path)
                freq_results['ghm_default'] = {
                    'history': ghm_history,
                    'best_val_accuracy': max(ghm_history['val_accuracy'])
                }
                
                # Analyze GHM stats if available
                if os.path.exists(ghm_stats_dir):
                    ghm_analysis_output = os.path.join(output_dir, f"{freq}_ghm_default_analysis")
                    analysis_results = analyze_single_run(ghm_stats_dir, ghm_plots_dir, ghm_analysis_output)
                    freq_results['ghm_default'].update(analysis_results)
        
        # GHM with different alpha values
        for alpha in [0.5, 0.9]:
            alpha_dir = find_latest_training_dirs(base_dir, f"*ghm_{freq}_alpha{alpha}*")
            if alpha_dir:
                alpha_history_path = os.path.join(alpha_dir, 'training_history.pkl')
                alpha_stats_dir = os.path.join(alpha_dir, 'ghm_stats')
                alpha_plots_dir = os.path.join(alpha_dir, 'plots')
                
                if os.path.exists(alpha_history_path):
                    alpha_history = load_training_history(alpha_history_path)
                    freq_results[f'ghm_alpha{alpha}'] = {
                        'history': alpha_history,
                        'best_val_accuracy': max(alpha_history['val_accuracy'])
                    }
                    
                    # Analyze GHM stats if available
                    if os.path.exists(alpha_stats_dir):
                        alpha_analysis_output = os.path.join(output_dir, f"{freq}_ghm_alpha{alpha}_analysis")
                        analysis_results = analyze_single_run(alpha_stats_dir, alpha_plots_dir, alpha_analysis_output)
                        freq_results[f'ghm_alpha{alpha}'].update(analysis_results)
        
        # GHM with different bin counts
        for bins in [5, 20]:
            bins_dir = find_latest_training_dirs(base_dir, f"*ghm_{freq}_bins{bins}*")
            if bins_dir:
                bins_history_path = os.path.join(bins_dir, 'training_history.pkl')
                bins_stats_dir = os.path.join(bins_dir, 'ghm_stats')
                bins_plots_dir = os.path.join(bins_dir, 'plots')
                
                if os.path.exists(bins_history_path):
                    bins_history = load_training_history(bins_history_path)
                    freq_results[f'ghm_bins{bins}'] = {
                        'history': bins_history,
                        'best_val_accuracy': max(bins_history['val_accuracy'])
                    }
                    
                    # Analyze GHM stats if available
                    if os.path.exists(bins_stats_dir):
                        bins_analysis_output = os.path.join(output_dir, f"{freq}_ghm_bins{bins}_analysis")
                        analysis_results = analyze_single_run(bins_stats_dir, bins_plots_dir, bins_analysis_output)
                        freq_results[f'ghm_bins{bins}'].update(analysis_results)
        
        results[freq] = freq_results
    
    # 2. Create comparison plots for each frequency
    for freq, freq_results in results.items():
        if not freq_results:
            continue
            
        # Compare training histories
        histories = []
        labels = []
        for param, param_results in freq_results.items():
            if 'history' in param_results:
                histories.append(param_results['history'])
                labels.append(param)
        
        if histories:
            comparison_plot = os.path.join(output_dir, f"{freq}_training_comparison.png")
            plot_training_comparison(histories, labels, comparison_plot)
        
        # Compare parameter impact for this frequency
        if len(freq_results) > 1:
            param_results = {param: results for param, results in freq_results.items()}
            parameter_plot = os.path.join(output_dir, f"{freq}_parameter_impact.png")
            create_parameter_impact_plot(param_results, parameter_plot)
    
    # 3. Compare frequencies
    if len(results) > 1:
        # For each frequency, get the best configuration
        freq_best = {}
        for freq, freq_results in results.items():
            if not freq_results:
                continue
                
            best_param = max(freq_results.items(), key=lambda x: x[1]['best_val_accuracy'] if 'best_val_accuracy' in x[1] else 0)
            freq_best[freq] = best_param[1]
        
        if freq_best:
            freq_comparison_plot = os.path.join(output_dir, "frequency_comparison.png")
            create_frequency_comparison_plot(freq_best, freq_comparison_plot)
    
    return results