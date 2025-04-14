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
    """Create a plot comparing performance across different frequencies."""
    freqs = list(results.keys())
    train_acc = [results[freq]['best_train_accuracy'] for freq in freqs]
    val_acc = [results[freq]['best_val_accuracy'] for freq in freqs]
    
    x = np.arange(len(freqs))
    width = 0.35
    
    plt.figure(figsize=(12, 8))
    plt.bar(x - width/2, train_acc, width, label='Training Accuracy', alpha=0.7)
    plt.bar(x + width/2, val_acc, width, label='Validation Accuracy', alpha=0.7)
    
    plt.title('GHM Performance Across Frequencies')
    plt.xlabel('Frequency')
    plt.ylabel('Best Accuracy (%)')
    plt.xticks(x, freqs)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (t_acc, v_acc) in enumerate(zip(train_acc, val_acc)):
        plt.text(i - width/2, t_acc + 0.5, f'{t_acc:.2f}%', ha='center', fontsize=9)
        plt.text(i + width/2, v_acc + 0.5, f'{v_acc:.2f}%', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return output_path

def main():
    """Main analysis function."""
    # Set up output directory
    output_base_dir = "ghm_analysis_results"
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 1. Find latest training runs
    save_base = "/Users/sbplab/Hank/angle_classification_deg6/saved_models"
    
    # Dictionary to store results for different configurations
    results = {}
    
    # Define configurations to analyze
    configs = {
        # Frequency comparison
        '500hz': {'pattern': 'model_checkpoints/plastic_500hz_ghm_*'},
        '1000hz': {'pattern': 'model_checkpoints/plastic_1000hz_ghm_*'},
        '3000hz': {'pattern': 'model_checkpoints/plastic_3000hz_ghm_*'},
        
        # GHM vs Standard for 1000Hz
        'standard_1000hz': {'pattern': 'model_checkpoints/plastic_1000hz_standard_*'},
        
        # Parameter variations for 1000Hz
        'bins5': {'pattern': 'model_checkpoints/plastic_1000hz_ghm_*_bins5_*'},
        'bins20': {'pattern': 'model_checkpoints/plastic_1000hz_ghm_*_bins20_*'},
        'alpha0.5': {'pattern': 'model_checkpoints/plastic_1000hz_ghm_*_alpha0.5_*'},
        'alpha0.9': {'pattern': 'model_checkpoints/plastic_1000hz_ghm_*_alpha0.9_*'},
    }
    
    # Process each configuration
    for config_name, config in configs.items():
        try:
            # Find latest training directory matching the pattern
            latest_dir = find_latest_training_dirs(save_base, config['pattern'])
            
            if not latest_dir:
                print(f"No matching directories found for {config_name}")
                continue
                
            # Load training history
            history_path = glob(os.path.join(latest_dir, "*_history_*.pkl"))[0]
            history = load_training_history(history_path)
            
            # Get best validation accuracy and corresponding epoch
            val_accuracies = history['val_accuracy']
            best_val_idx = np.argmax(val_accuracies)
            best_val_accuracy = val_accuracies[best_val_idx]
            best_val_epoch = history['epoch'][best_val_idx]
            
            # Get training accuracy at the same epoch
            train_accuracy = history['train_accuracy'][best_val_idx]
            
            # Store results
            results[config_name] = {
                'history': history,
                'history_path': history_path,
                'model_dir': latest_dir,
                'best_val_accuracy': best_val_accuracy,
                'best_val_epoch': best_val_epoch,
                'best_train_accuracy': train_accuracy
            }
            
            print(f"{config_name}: Best val accuracy {best_val_accuracy:.2f}% at epoch {best_val_epoch}")
            
            # For GHM runs, perform additional GHM-specific analysis
            if 'standard' not in config_name:
                stats_dir_pattern = latest_dir.replace('model_checkpoints', 'stats')
                stats_dir = glob(stats_dir_pattern)[0] if glob(stats_dir_pattern) else None
                
                if stats_dir:
                    plots_dir = latest_dir.replace('model_checkpoints', 'plots')
                    analysis_output_dir = os.path.join(output_base_dir, f"{config_name}_analysis")
                    
                    ghm_results = analyze_single_run(stats_dir, plots_dir, analysis_output_dir)
                    results[config_name]['ghm_analysis'] = ghm_results
        
        except Exception as e:
            print(f"Error processing {config_name}: {str(e)}")
    
    # Create comparison plots
    
    # 1. GHM vs Standard Loss comparison
    if 'standard_1000hz' in results and '1000hz' in results:
        ghm_vs_std_histories = [
            results['1000hz']['history'],
            results['standard_1000hz']['history']
        ]
        ghm_vs_std_labels = ['GHM Loss', 'Standard Loss']
        
        ghm_vs_std_plot = plot_training_comparison(
            ghm_vs_std_histories, 
            ghm_vs_std_labels,
            os.path.join(output_base_dir, "ghm_vs_standard_comparison.png")
        )
        print(f"Created GHM vs Standard comparison plot: {ghm_vs_std_plot}")
    
    # 2. Frequency comparison
    freq_configs = ['500hz', '1000hz', '3000hz']
    if all(freq in results for freq in freq_configs):
        freq_results = {freq: results[freq] for freq in freq_configs}
        freq_plot = create_frequency_comparison_plot(
            freq_results,
            os.path.join(output_base_dir, "frequency_comparison.png")
        )
        print(f"Created frequency comparison plot: {freq_plot}")
    
    # 3. GHM Parameter impact
    param_configs = {
        'bins=5, alpha=0.75': 'bins5',
        'bins=10, alpha=0.75': '1000hz',
        'bins=20, alpha=0.75': 'bins20',
        'bins=10, alpha=0.5': 'alpha0.5',
        'bins=10, alpha=0.9': 'alpha0.9'
    }
    
    if all(config in results for config in param_configs.values()):
        param_results = {
            name: results[config] for name, config in param_configs.items()
        }
        param_plot = create_parameter_impact_plot(
            param_results,
            os.path.join(output_base_dir, "parameter_impact.png")
        )
        print(f"Created parameter impact plot: {param_plot}")
    
    # 4. Create a summary report
    with open(os.path.join(output_base_dir, "analysis_summary.txt"), "w") as f:
        f.write("# GHM Training and Analysis Results\n\n")
        
        f.write("## Best Validation Accuracies\n\n")
        for config_name, result in results.items():
            f.write(f"{config_name}: {result['best_val_accuracy']:.2f}% at epoch {result['best_val_epoch']}\n")
        
        f.write("\n## Configuration Comparisons\n\n")
        
        # GHM vs Standard
        if 'standard_1000hz' in results and '1000hz' in results:
            ghm_acc = results['1000hz']['best_val_accuracy']
            std_acc = results['standard_1000hz']['best_val_accuracy']
            diff = ghm_acc - std_acc
            f.write(f"GHM vs Standard (1000Hz): {diff:.2f}% difference\n")
        
        # Frequency impact
        if all(freq in results for freq in freq_configs):
            best_freq = max(freq_configs, key=lambda x: results[x]['best_val_accuracy'])
            f.write(f"\nBest frequency: {best_freq} ({results[best_freq]['best_val_accuracy']:.2f}%)\n")
        
        # Parameter impact
        if all(config in results for config in param_configs.values()):
            best_param = max(param_configs.items(), key=lambda x: results[x[1]]['best_val_accuracy'])
            f.write(f"\nBest GHM parameters: {best_param[0]} ({results[best_param[1]]['best_val_accuracy']:.2f}%)\n")
        
        f.write("\n## Conclusions\n\n")
        f.write("1. [Add conclusions based on the analysis results]\n")
        f.write("2. [Add recommendations for future work]\n")
    
    print(f"Analysis complete. Results saved to {output_base_dir}")

if __name__ == "__main__":
    main() 