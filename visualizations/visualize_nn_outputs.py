"""
Neural Network Output Visualization Tool
Features:
- Load trained CNN models
- Extract final output scores from the network
- Visualize output distribution by angle
- Compare outputs across different data points
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # Set backend
from matplotlib.backends.backend_agg import FigureCanvasAgg
import sys

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import required modules
import config
from datasets import SpectrogramDatasetWithMaterial
# from simple_cnn_models import ResNetAudioRanker
from simple_cnn_models_native import SimpleCNNAudioRanker as ResNetAudioRanker

# Try different import approaches


def get_angle_from_class(class_name):
    """Extract angle value from class name"""
    # Assuming class name format is 'degXXX', where XXX is the angle value
    try:
        return int(class_name[3:])  # Extract number after 'deg'
    except ValueError:
        return 0

def extract_nn_outputs(model, dataset, device):
    """
    Extract final output values from the neural network for each data point
    Returns: output values, class labels, angle values
    """
    model.eval()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_outputs = []
    all_labels = []
    all_angles = []
    
    with torch.no_grad():
        for data, labels in dataloader:
            # Move data to device
            data = data.to(device)
            
            # Get model outputs
            outputs = model(data).squeeze().cpu().numpy()
            
            # Convert labels to angles
            angles = [get_angle_from_class(dataset.classes[label.item()]) for label in labels]
            
            # Save results
            all_outputs.extend(outputs)
            all_labels.extend(labels.numpy())
            all_angles.extend(angles)
    
    return np.array(all_outputs), np.array(all_labels), np.array(all_angles)

def plot_outputs_by_angle(outputs, angles, save_path):
    """Plot NN outputs grouped by angle"""
    plt.figure(figsize=(14, 8))
    
    # Create violin plot
    # Order angles
    unique_angles = sorted(np.unique(angles))
    
    # Create data for plotting
    data_by_angle = [outputs[angles == angle] for angle in unique_angles]
    
    # Plot violin plots
    parts = plt.violinplot(data_by_angle, showmeans=True, showmedians=True)
    
    # Set colors
    for pc in parts['bodies']:
        pc.set_facecolor('#3182bd')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    
    # Add scatter points for individual data
    for i, angle in enumerate(unique_angles):
        # Add jitter to x-position
        x = np.full_like(outputs[angles == angle], i + 1)
        x = x + np.random.normal(0, 0.05, size=len(x))
        plt.scatter(x, outputs[angles == angle], alpha=0.6, color='#636363', s=20)
    
    # Set x-axis ticks and labels
    plt.xticks(np.arange(1, len(unique_angles) + 1), [f"{angle}°" for angle in unique_angles])
    
    plt.title('Neural Network Output Distribution by Angle', fontsize=16)
    plt.xlabel('Angle (degrees)', fontsize=14)
    plt.ylabel('Network Output Value', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Output by angle chart saved to: {save_path}")

def plot_output_heatmap(outputs, angles, save_path):
    """Create a heatmap showing output distribution by angle"""
    # Create a 2D histogram of angles vs outputs
    unique_angles = sorted(np.unique(angles))
    
    # Define output bins
    min_output = np.min(outputs)
    max_output = np.max(outputs)
    output_bins = np.linspace(min_output, max_output, 20)
    
    # Initialize histogram matrix
    hist_matrix = np.zeros((len(unique_angles), len(output_bins) - 1))
    
    # Fill histogram
    for i, angle in enumerate(unique_angles):
        angle_outputs = outputs[angles == angle]
        hist, _ = np.histogram(angle_outputs, bins=output_bins)
        hist_matrix[i, :] = hist / np.max(hist)  # Normalize
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(hist_matrix, cmap="YlGnBu", 
                     xticklabels=[f"{output_bins[i]:.2f}" for i in range(0, len(output_bins) - 1, 3)],
                     yticklabels=[f"{angle}°" for angle in unique_angles])
    
    # Adjust x-ticks
    plt.xticks(np.arange(0, len(output_bins) - 1, 3))
    
    plt.title('Output Value Distribution Heatmap by Angle', fontsize=16)
    plt.xlabel('Network Output Value', fontsize=14)
    plt.ylabel('Angle (degrees)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Output heatmap saved to: {save_path}")

def plot_angle_vs_output_scatter(outputs, angles, save_path):
    """Create a scatter plot of angles vs. network outputs"""
    plt.figure(figsize=(12, 8))
    
    # Scatter plot with trend line
    plt.scatter(angles, outputs, alpha=0.7, s=60, c=outputs, cmap='viridis')
    
    # Add trend line
    z = np.polyfit(angles, outputs, 1)
    p = np.poly1d(z)
    plt.plot(sorted(np.unique(angles)), p(sorted(np.unique(angles))), 
             'r--', linewidth=2, label=f'Trend: y={z[0]:.4f}x+{z[1]:.4f}')
    
    plt.title('Network Output Values vs. Angle', fontsize=16)
    plt.xlabel('Angle (degrees)', fontsize=14)
    plt.ylabel('Network Output Value', fontsize=14)
    plt.grid(alpha=0.3)
    plt.colorbar(label='Output Value')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Angle vs. output scatter plot saved to: {save_path}")

def plot_output_histogram(outputs, save_path):
    """Create a histogram of all network output values"""
    plt.figure(figsize=(12, 6))
    
    # Create histogram with density curve
    sns.histplot(outputs, bins=30, kde=True)
    
    # Add statistics
    mean_output = np.mean(outputs)
    median_output = np.median(outputs)
    std_output = np.std(outputs)
    
    plt.axvline(mean_output, color='r', linestyle='--', 
                label=f'Mean: {mean_output:.4f}')
    plt.axvline(median_output, color='g', linestyle='-.', 
                label=f'Median: {median_output:.4f}')
    
    plt.title(f'Distribution of Network Output Values (σ={std_output:.4f})', fontsize=16)
    plt.xlabel('Network Output Value', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Output histogram saved to: {save_path}")

def plot_multi_angle_comparison(outputs1, angles1, outputs2, angles2, material1, material2, frequency, save_path):
    """Create a scatter plot comparing two datasets with different angle intervals"""
    plt.figure(figsize=(12, 8))
    
    # 為兩個數據集使用不同的標記和顏色
    plt.scatter(angles1, outputs1, alpha=0.7, s=100, marker='o', 
               label=f'{material1} (36° intervals)', c='#1f77b4')
    plt.scatter(angles2, outputs2, alpha=0.7, s=80, marker='^', 
               label=f'{material2} (18° intervals)', c='#ff7f0e')
    
    # 為每個數據集添加趨勢線
    z1 = np.polyfit(angles1, outputs1, 1)
    p1 = np.poly1d(z1)
    plt.plot(sorted(np.unique(angles1)), p1(sorted(np.unique(angles1))), 
             '--', color='#1f77b4', linewidth=2, 
             label=f'{material1} trend: y={z1[0]:.4f}x+{z1[1]:.4f}')
    
    z2 = np.polyfit(angles2, outputs2, 1)
    p2 = np.poly1d(z2)
    plt.plot(sorted(np.unique(angles2)), p2(sorted(np.unique(angles2))), 
             '--', color='#ff7f0e', linewidth=2, 
             label=f'{material2} trend: y={z2[0]:.4f}x+{z2[1]:.4f}')
    
    # 添加垂直線標記角度間隔 (可選)
    # angle_range = np.arange(0, 181, 36)
    # for angle in angle_range:
    #    plt.axvline(x=angle, color='#1f77b4', alpha=0.1, linestyle='-')
    
    # angle_range_2 = np.arange(0, 181, 18)
    # for angle in angle_range_2:
    #    plt.axvline(x=angle, color='#ff7f0e', alpha=0.1, linestyle=':')
    
    plt.title(f'Network Output Values vs. Angle Comparison\n{frequency}', fontsize=16)
    plt.xlabel('Angle (degrees)', fontsize=14)
    plt.ylabel('Network Output Value', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 調整x軸範圍
    plt.xlim(-5, 185)
    
    # 添加統計信息到圖表內部
    stats_text = f"{material1}: μ={np.mean(outputs1):.2f}, σ={np.std(outputs1):.2f}\n{material2}: μ={np.mean(outputs2):.2f}, σ={np.std(outputs2):.2f}"
    plt.text(0.02, 0.02, stats_text, transform=plt.gca().transAxes, 
            bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
    
    # 調整圖例位置
    plt.legend(loc='upper right', fontsize=10)
    
    # 設置x軸刻度
    major_ticks = np.arange(0, 181, 36)
    minor_ticks = np.arange(0, 181, 18)
    plt.gca().set_xticks(major_ticks)
    plt.gca().set_xticks(minor_ticks, minor=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Comparison plot saved to: {save_path}")

def main():
    print("Starting Neural Network Output visualization process...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get latest model file
    model_files = [f for f in os.listdir(config.SAVE_DIR) if f.startswith('simple_cnn_') and f.endswith('.pt')]
    if not model_files:
        print("Could not find model files!")
        return

    # Select the latest model file
    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(config.SAVE_DIR, x)))
    model_path = "/Users/sbplab/Hank/angle_classification_deg6/saved_models/resnet18_plastic_1000hz_best_20250401_201457.pt"
    print(f"\nLoading model: {latest_model}")

    # Parse frequency and material information from filename
    parts = latest_model.split('_')
    try:
        material = "plastic"
        frequency = "1000hz"
    except IndexError:
        print("Error parsing model filename. Using default values.")
        material = "plastic"
        frequency = "1000hz"
    
    # Load first dataset (original, 36度間隔)
    dataset1 = SpectrogramDatasetWithMaterial(
        os.path.join(config.DATA_ROOT, "step_018_sliced"),
        ["deg000", "deg036", "deg072", "deg108", "deg144", "deg180"],
        config.SEQ_NUMS,
        frequency,
        material
    )
    
    if len(dataset1) == 0:
        print("First dataset is empty!")
        return
    
    print(f"First dataset loaded with {len(dataset1)} samples")

    # Load second dataset (18度間隔)
    second_material = "plastic"
    
    dataset2 = SpectrogramDatasetWithMaterial(
        os.path.join(config.DATA_ROOT, "step_018_sliced"),
        ["deg000", "deg018", "deg036", "deg054", "deg072", "deg090", "deg108", "deg126", "deg144", "deg162", "deg180"],
        config.SEQ_NUMS,
        frequency,
        material
    )
    
    if len(dataset2) == 0:
        print("Second dataset is empty!")
        return
    
    print(f"Second dataset loaded with {len(dataset2)} samples")
    
    # Load model
    model = ResNetAudioRanker(n_freqs=dataset1.data.shape[2])
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print("Extracting network output values...")
    outputs1, labels1, angles1 = extract_nn_outputs(model, dataset1, device)
    outputs2, labels2, angles2 = extract_nn_outputs(model, dataset2, device)
    
    # Create save directory
    plots_dir = os.path.join(config.SAVE_DIR, 'nn_output_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate visualizations
    print("Generating visualization charts...")
    
    # 使用新的視覺化函數
    scatter_path = os.path.join(plots_dir, f'angle_vs_output_comparison_{material}_{second_material}_{frequency}.png')
    plot_multi_angle_comparison(outputs1, angles1, outputs2, angles2, 
                              material, second_material, frequency, scatter_path)
    
    print("\nVisualization complete! Charts saved to:", plots_dir)
    print(f"Processed {len(outputs1)} data points from first dataset (36° intervals)")
    print(f"Processed {len(outputs2)} data points from second dataset (18° intervals)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc() 