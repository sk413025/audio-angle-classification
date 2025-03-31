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

# IMPORT_CONFIG_COMPLETE
# Add the parent directory to the Python path
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

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

def main():
    print("Starting Neural Network Output visualization process...")
    device = config.DEVICE
    print(f"Using device: {device}")
    
    # Let user select frequency
    available_frequencies = ['500hz', '1000hz', '3000hz']
    print("\nAvailable frequencies:")
    for i, freq in enumerate(available_frequencies):
        print(f"{i+1}. {freq}")
    
    while True:
        try:
            choice = int(input("\nSelect frequency (1-3): "))
            if 1 <= choice <= 3:
                selected_freq = available_frequencies[choice-1]
                break
            else:
                print("Invalid choice, please enter a number between 1-3")
        except ValueError:
            print("Please enter a valid number")
    
    # Get latest model file for the specified frequency
    model_files = [
        f for f in os.listdir(config.SAVE_DIR) 
        if f.startswith('resnet18_') 
        and selected_freq in f 
        and f.endswith('.pt')
    ]
    
    if not model_files:
        print(f"Could not find model files for frequency {selected_freq}!")
        return
    
    # Select the latest model file
    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(config.SAVE_DIR, x)))
    model_path = os.path.join(config.SAVE_DIR, latest_model)
    
    print(f"\nLoading model: {latest_model}")
    
    # Parse frequency and material information from filename
    parts = latest_model.split('_')
    try:
        # material = parts[2]
        # frequency = parts[3]
        material = "plastic"
        frequency = "500hz"
    except IndexError:
        print("Error parsing model filename. Using default values.")
        material = "unknown"
        frequency = "unknown"
    
    # Load dataset
    # import ipdb; ipdb.set_trace()
    dataset = SpectrogramDatasetWithMaterial(
        config.DATA_ROOT,
        config.CLASSES,
        config.SEQ_NUMS,
        frequency,
        material
    )
    
    if len(dataset) == 0:
        print("Dataset is empty!")
        return
    
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Load model
    model = SimpleCNNAudioRanker(n_freqs=dataset.data.shape[2])
    # model = ResNetAudioRanker(n_freqs=dataset.data.shape[2])
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print("Extracting network output values...")
    outputs, labels, angles = extract_nn_outputs(model, dataset, device)
    
    # Create save directory
    plots_dir = os.path.join(config.SAVE_DIR, 'nn_output_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate visualizations
    print("Generating visualization charts...")
    
    # 1. Output by angle violin plot
    # violin_path = os.path.join(plots_dir, f'output_by_angle_{material}_{frequency}.png')
    # plot_outputs_by_angle(outputs, angles, violin_path)
    
    # # 2. Output distribution heatmap
    # heatmap_path = os.path.join(plots_dir, f'output_heatmap_{material}_{frequency}.png')
    # plot_output_heatmap(outputs, angles, heatmap_path)
    
    # # 3. Angle vs. output scatter plot
    scatter_path = os.path.join(plots_dir, f'angle_vs_output_{material}_{frequency}.png')
    plot_angle_vs_output_scatter(outputs, angles, scatter_path)
    
    # # 4. Output histogram
    # hist_path = os.path.join(plots_dir, f'output_histogram_{material}_{frequency}.png')
    # plot_output_histogram(outputs, hist_path)
    
    print("\nVisualization complete! Charts saved to:", plots_dir)
    print(f"Processed {len(outputs)} data points")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc() 