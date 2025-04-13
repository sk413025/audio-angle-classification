import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
import glob
from pathlib import Path

def analyze_ghm_training_results(frequency='3000hz', material='plastic', timestamp='20250411_222821'):
    """
    Analyze GHM training results and create a comprehensive report
    
    Args:
        frequency: The frequency used for training ('500hz', '1000hz', '3000hz')
        material: The material used ('plastic')
        timestamp: The timestamp of the training run to analyze
    """
    print(f"Analyzing GHM training results for {material}_{frequency} (timestamp: {timestamp})")
    
    # Create report directory
    report_dir = os.path.join('ghm_analysis_report', f"{material}_{frequency}_{timestamp}")
    os.makedirs(report_dir, exist_ok=True)
    
    # Create subdirectories
    gradient_plots_dir = os.path.join(report_dir, 'gradient_plots')
    os.makedirs(gradient_plots_dir, exist_ok=True)
    
    stats_analysis_dir = os.path.join(report_dir, 'stats_analysis')
    os.makedirs(stats_analysis_dir, exist_ok=True)
    
    # Paths to data
    ghm_plots_dir = os.path.join('./saved_models/ghm_plots', f"{material}_{frequency}")
    ghm_stats_dir = os.path.join('./saved_models/ghm_stats', f"{material}_{frequency}")
    model_checkpoints_dir = os.path.join('./saved_models/model_checkpoints', f"{material}_{frequency}_ghm")
    
    # Find the training history file
    history_files = glob.glob(os.path.join(model_checkpoints_dir, f"training_history_ghm_{timestamp}.pt"))
    
    if not history_files:
        print(f"No training history found for timestamp {timestamp}")
        return
    
    history_file = history_files[0]
    print(f"Using training history file: {history_file}")
    
    # Load training history
    try:
        training_history = torch.load(history_file)
        print("Training history loaded successfully")
    except Exception as e:
        print(f"Error loading training history: {e}")
        return
    
    # Create markdown report
    report_file = os.path.join(report_dir, 'analysis_report.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# GHM Training Analysis Report\n\n")
        f.write(f"## Training Parameters\n\n")
        f.write(f"- **Material:** {material}\n")
        f.write(f"- **Frequency:** {frequency}\n")
        f.write(f"- **Timestamp:** {timestamp}\n")
        f.write(f"- **Random Seed:** {training_history.get('random_seed', 'Not specified')}\n")
        
        # Add GHM parameters
        ghm_params = training_history.get('ghm_params', {})
        f.write(f"- **GHM Bins:** {ghm_params.get('bins', 'Not specified')}\n")
        f.write(f"- **GHM Alpha:** {ghm_params.get('alpha', 'Not specified')}\n")
        f.write(f"- **GHM Margin:** {ghm_params.get('margin', 'Not specified')}\n")
        f.write(f"- **Total Epochs:** {max(training_history.get('epoch', [0]))}\n\n")
        
        # Training and validation performance
        f.write(f"## Training and Validation Performance\n\n")
        
        # Find comparison plot and copy to report
        comparison_plots = glob.glob(os.path.join(ghm_plots_dir, f"training_comparison_{timestamp}.png"))
        if comparison_plots:
            comparison_plot = comparison_plots[0]
            dest_file = os.path.join(report_dir, os.path.basename(comparison_plot))
            shutil.copy2(comparison_plot, dest_file)
            f.write(f"![Training Performance]({os.path.basename(comparison_plot)})\n\n")
            f.write(f"*Figure 1: Training and validation performance comparing original loss vs GHM loss*\n\n")
        
        # Best performance analysis
        epochs = training_history.get('epoch', [])
        train_losses = training_history.get('train_loss', [])
        train_ghm_losses = training_history.get('train_ghm_loss', [])
        val_losses = training_history.get('val_loss', [])
        val_ghm_losses = training_history.get('val_ghm_loss', [])
        train_accuracies = training_history.get('train_accuracy', [])
        val_accuracies = training_history.get('val_accuracy', [])
        
        if val_accuracies:
            best_val_acc_idx = np.argmax(val_accuracies)
            best_val_acc = val_accuracies[best_val_acc_idx]
            best_val_acc_epoch = epochs[best_val_acc_idx]
            
            f.write(f"### Best Model Performance\n\n")
            f.write(f"- **Best Validation Accuracy:** {best_val_acc:.2f}% (Epoch {best_val_acc_epoch})\n")
            f.write(f"- **Corresponding Training Accuracy:** {train_accuracies[best_val_acc_idx]:.2f}%\n")
            f.write(f"- **Original Validation Loss:** {val_losses[best_val_acc_idx]:.4f}\n")
            f.write(f"- **GHM Validation Loss:** {val_ghm_losses[best_val_acc_idx]:.4f}\n\n")
        
        # Gradient distribution analysis
        f.write(f"## Gradient Distribution Analysis\n\n")
        f.write(f"The following plots show the gradient distribution at different stages of training.\n")
        f.write(f"This helps visualize how GHM loss reshapes the gradient to focus on more informative samples.\n\n")
        
        # Find and categorize gradient plots by epoch
        gradient_plots = glob.glob(os.path.join(ghm_plots_dir, f"epoch*_{timestamp}.png"))
        
        # Sort and organize plots by epoch
        epoch_plots = {}
        for plot_path in gradient_plots:
            base_name = os.path.basename(plot_path)
            if base_name.startswith('epoch'):
                parts = base_name.split('_')
                epoch_num = int(parts[0].replace('epoch', ''))
                if epoch_num not in epoch_plots:
                    epoch_plots[epoch_num] = []
                epoch_plots[epoch_num].append(plot_path)
        
        # Sample epochs to include in report (beginning, middle, end)
        if epoch_plots:
            all_epochs = sorted(epoch_plots.keys())
            sample_epochs = []
            
            # Beginning
            if all_epochs:
                sample_epochs.append(all_epochs[0])
            
            # Middle
            if len(all_epochs) > 2:
                sample_epochs.append(all_epochs[len(all_epochs) // 2])
            
            # End
            if len(all_epochs) > 1:
                sample_epochs.append(all_epochs[-1])
            
            f.write(f"### Gradient Evolution During Training\n\n")
            
            for idx, epoch in enumerate(sample_epochs):
                if epoch_plots.get(epoch):
                    # Copy the first batch's gradient plot
                    for plot_path in epoch_plots[epoch]:
                        if 'batch0' in plot_path:
                            base_name = os.path.basename(plot_path)
                            dest_file = os.path.join(gradient_plots_dir, base_name)
                            shutil.copy2(plot_path, dest_file)
                            rel_path = os.path.join('gradient_plots', base_name)
                            f.write(f"![Epoch {epoch} Gradient Distribution]({rel_path})\n\n")
                            f.write(f"*Figure {idx+2}: Gradient distribution at Epoch {epoch}*\n\n")
                            break
        
        # GHM statistics analysis
        f.write(f"## GHM Statistics Analysis\n\n")
        
        # Find and analyze a few sample GHM statistics files
        stats_files = glob.glob(os.path.join(ghm_stats_dir, f"epoch*_{timestamp}.npy"))
        
        if stats_files:
            # Sample stats from beginning, middle, and end
            sample_stats_files = []
            
            # Find groups based on epoch number
            epoch_stats = {}
            for stats_path in stats_files:
                base_name = os.path.basename(stats_path)
                if base_name.startswith('epoch'):
                    parts = base_name.split('_')
                    epoch_num = int(parts[0].replace('epoch', ''))
                    if epoch_num not in epoch_stats:
                        epoch_stats[epoch_num] = []
                    epoch_stats[epoch_num].append(stats_path)
            
            all_stat_epochs = sorted(epoch_stats.keys())
            
            if all_stat_epochs:
                # Beginning
                if all_stat_epochs:
                    if epoch_stats.get(all_stat_epochs[0]):
                        sample_stats_files.append(epoch_stats[all_stat_epochs[0]][0])
                
                # Middle
                if len(all_stat_epochs) > 2:
                    mid_epoch = all_stat_epochs[len(all_stat_epochs) // 2]
                    if epoch_stats.get(mid_epoch):
                        sample_stats_files.append(epoch_stats[mid_epoch][0])
                
                # End
                if len(all_stat_epochs) > 1:
                    if epoch_stats.get(all_stat_epochs[-1]):
                        sample_stats_files.append(epoch_stats[all_stat_epochs[-1]][0])
            
            # Plot and analyze the distributions
            for idx, stats_file in enumerate(sample_stats_files):
                try:
                    stats_data = np.load(stats_file, allow_pickle=True).item()  # Load as dictionary
                    epoch_num = int(os.path.basename(stats_file).split('_')[0].replace('epoch', ''))
                    
                    # Extract bin counts for plotting
                    bin_counts = stats_data.get('bin_counts', [])
                    bin_edges = stats_data.get('bin_edges', np.arange(len(bin_counts) + 1))
                    
                    # Create a plot
                    plt.figure(figsize=(10, 6))
                    plt.bar(range(len(bin_counts)), bin_counts)
                    plt.xlabel('Bins')
                    plt.ylabel('Gradient Count')
                    plt.title(f'GHM Gradient Bin Distribution - Epoch {epoch_num}')
                    plt.grid(True, alpha=0.3)
                    
                    # Save the plot
                    plot_filename = f'ghm_stats_epoch_{epoch_num}.png'
                    plot_path = os.path.join(stats_analysis_dir, plot_filename)
                    plt.savefig(plot_path)
                    plt.close()
                    
                    # Add to report
                    rel_path = os.path.join('stats_analysis', plot_filename)
                    f.write(f"![Epoch {epoch_num} GHM Stats]({rel_path})\n\n")
                    f.write(f"*Figure {len(sample_epochs)+idx+2}: GHM gradient bin distribution at Epoch {epoch_num}*\n\n")
                    
                    # Calculate statistics
                    total_gradients = sum(bin_counts)
                    if total_gradients > 0:
                        bin_percentages = [count / total_gradients * 100 for count in bin_counts]
                        max_bin = bin_counts.index(max(bin_counts)) if bin_counts else 0
                        min_bin = bin_counts.index(min(bin_counts)) if bin_counts else 0
                        
                        f.write(f"**GHM Statistics for Epoch {epoch_num}:**\n\n")
                        f.write(f"- Total gradients: {total_gradients}\n")
                        f.write(f"- Mean gradient: {stats_data.get('mean_gradient', 'N/A'):.4f}\n")
                        f.write(f"- Median gradient: {stats_data.get('median_gradient', 'N/A'):.4f}\n")
                        f.write(f"- Min gradient: {stats_data.get('min_gradient', 'N/A'):.4f}\n")
                        f.write(f"- Max gradient: {stats_data.get('max_gradient', 'N/A'):.4f}\n")
                        f.write(f"- Std deviation of gradients: {stats_data.get('std_gradient', 'N/A'):.4f}\n")
                        f.write(f"- Most populated bin: {max_bin} ({bin_percentages[max_bin]:.2f}%)\n")
                        f.write(f"- Least populated bin: {min_bin} ({bin_percentages[min_bin]:.2f}%)\n")
                        f.write(f"- Gradient distribution evenness: {np.std(bin_percentages):.2f}% (standard deviation)\n\n")
                
                except Exception as e:
                    print(f"Error analyzing stats file {stats_file}: {e}")
        
        # Conclusion
        f.write(f"## Conclusion\n\n")
        
        # Calculate overall improvement
        if val_accuracies and len(val_accuracies) > 1:
            initial_acc = val_accuracies[0]
            final_acc = val_accuracies[-1]
            improvement = final_acc - initial_acc
            
            f.write(f"### Overall Training Results\n\n")
            f.write(f"- **Initial Validation Accuracy:** {initial_acc:.2f}%\n")
            f.write(f"- **Final Validation Accuracy:** {final_acc:.2f}%\n")
            f.write(f"- **Overall Improvement:** {improvement:.2f}%\n\n")
            
            if improvement > 0:
                f.write(f"The model shows positive improvement during training with GHM loss.\n")
            else:
                f.write(f"The model shows stability but limited accuracy gain with GHM loss.\n")
            
            # Compare GHM loss with original loss
            if val_losses and val_ghm_losses and len(val_losses) > 1 and len(val_ghm_losses) > 1:
                orig_loss_improvement = val_losses[0] - val_losses[-1]
                ghm_loss_improvement = val_ghm_losses[0] - val_ghm_losses[-1]
                
                f.write(f"\n**Loss Analysis:**\n\n")
                f.write(f"- Original loss reduction: {orig_loss_improvement:.4f}\n")
                f.write(f"- GHM loss reduction: {ghm_loss_improvement:.4f}\n\n")
        
        f.write(f"### Effectiveness of GHM\n\n")
        f.write(f"The Gradient Harmonizing Mechanism (GHM) was applied to balance the training process by reshaping the gradient distributions. ")
        f.write(f"Based on the visualizations and statistics, we can observe how the gradient distribution evolved throughout training.\n\n")
        
        f.write(f"GHM helps balance the contribution of easy vs. hard examples, potentially leading to more robust model training ")
        f.write(f"especially in cases where the dataset contains imbalanced examples or difficulty levels.\n\n")
        
        f.write(f"The final model checkpoint can be found at: \n`{model_checkpoints_dir}/model_ghm_epoch_{max(epochs)}_{timestamp}.pt`\n")
    
    print(f"Analysis report generated at: {report_file}")
    return report_file

if __name__ == "__main__":
    # Set frequency and material
    frequency = '3000hz'
    material = 'plastic'
    
    # Find the latest timestamp from the training history files
    model_checkpoints_dir = os.path.join('./saved_models/model_checkpoints', f"{material}_{frequency}_ghm")
    history_files = glob.glob(os.path.join(model_checkpoints_dir, "training_history_ghm_*.pt"))
    
    latest_timestamp = None
    if history_files:
        for file in history_files:
            # Extract the full timestamp part (e.g., "20250411_222821")
            file_name = os.path.basename(file)
            if 'training_history_ghm_' in file_name:
                # Extract everything after "training_history_ghm_" and before ".pt"
                timestamp = file_name.replace('training_history_ghm_', '').replace('.pt', '')
                if latest_timestamp is None or timestamp > latest_timestamp:
                    latest_timestamp = timestamp
    
    if latest_timestamp:
        print(f"Found latest training run timestamp: {latest_timestamp}")
        analyze_ghm_training_results(frequency, material, latest_timestamp)
    else:
        print("No training history files found. Please specify a timestamp manually.") 