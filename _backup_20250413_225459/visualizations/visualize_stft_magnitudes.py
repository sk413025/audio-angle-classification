"""
STFT Magnitude Visualization Tool
Features:
- Load spectrogram dataset
- Visualize STFT magnitude distributions
- Compare spectrograms across different angles
- Analyze frequency patterns in the data
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

# Import from parent directory


def get_angle_from_class(class_name):
    """Extract angle value from class name"""
    # Assuming class name format is 'degXXX', where XXX is the angle value
    try:
        return int(class_name[3:])  # Extract number after 'deg'
    except ValueError:
        return 0

def plot_spectrogram(spec, title, save_path, vmin=None, vmax=None):
    """Plot a single spectrogram"""
    plt.figure(figsize=(10, 6))
    
    # Transpose to get frequency on y-axis and time on x-axis
    plt.imshow(spec.T, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    
    plt.colorbar(label='Magnitude')
    plt.title(title, fontsize=16)
    plt.xlabel('Time Frame', fontsize=14)
    plt.ylabel('Frequency Bin', fontsize=14)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Spectrogram saved to: {save_path}")

def plot_average_spectrograms_by_angle(dataset, angles, save_dir):
    """Plot average spectrograms for each angle"""
    # Group spectrograms by angle
    angle_to_specs = {}
    for i, angle in enumerate(angles):
        if angle not in angle_to_specs:
            angle_to_specs[angle] = []
        angle_to_specs[angle].append(dataset.data[i])
    
    # Calculate average spectrograms per angle
    for angle, specs in angle_to_specs.items():
        avg_spec = np.mean(specs, axis=0)
        
        # Plot the average spectrogram
        title = f'Average Spectrogram for Angle {angle}°'
        save_path = os.path.join(save_dir, f'avg_spec_angle_{angle}.png')
        plot_spectrogram(avg_spec, title, save_path)

def plot_spectrogram_grid(dataset, angles, material, frequency, save_path, samples_per_angle=3):
    """Plot a grid of example spectrograms for each angle"""
    unique_angles = sorted(np.unique(angles))
    
    # Determine grid dimensions
    n_angles = len(unique_angles)
    n_cols = samples_per_angle
    n_rows = n_angles
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3))
    
    # Find global min and max for consistent colormap
    all_specs = dataset.data
    vmin, vmax = np.min(all_specs), np.max(all_specs)
    
    # For each angle, plot sample spectrograms
    for i, angle in enumerate(unique_angles):
        # Get indices of samples with this angle
        angle_indices = np.where(angles == angle)[0]
        
        # Select a subset of samples
        if len(angle_indices) > samples_per_angle:
            # Randomly sample
            selected_indices = np.random.choice(angle_indices, samples_per_angle, replace=False)
        else:
            # Use all available with possible repetition
            selected_indices = np.random.choice(angle_indices, samples_per_angle, replace=True)
        
        # Plot each selected sample
        for j, idx in enumerate(selected_indices):
            spec = dataset.data[idx]
            ax = axes[i, j]
            im = ax.imshow(spec.T, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_title(f'Angle {angle}° (Sample {j+1})')
            
            # Only add axis labels for edge plots
            if i == n_rows - 1:
                ax.set_xlabel('Time Frame')
            if j == 0:
                ax.set_ylabel('Frequency Bin')
            
            # Remove ticks for cleaner appearance
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Add colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Magnitude')
    
    # Add overall title
    fig.suptitle(f'Spectrograms by Angle ({material}, {frequency})', fontsize=16, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Spectrogram grid saved to: {save_path}")

def plot_magnitude_distribution(dataset, save_path):
    """Plot distribution of STFT magnitude values"""
    # Flatten all spectrogram data
    all_magnitudes = dataset.data.flatten()
    
    plt.figure(figsize=(12, 6))
    
    # Create histogram with density curve
    sns.histplot(all_magnitudes, bins=100, kde=True)
    
    # Add statistics
    mean_mag = np.mean(all_magnitudes)
    median_mag = np.median(all_magnitudes)
    std_mag = np.std(all_magnitudes)
    
    plt.axvline(mean_mag, color='r', linestyle='--', 
                label=f'Mean: {mean_mag:.4f}')
    plt.axvline(median_mag, color='g', linestyle='-.', 
                label=f'Median: {median_mag:.4f}')
    
    plt.title(f'Distribution of STFT Magnitude Values (σ={std_mag:.4f})', fontsize=16)
    plt.xlabel('Magnitude', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Magnitude distribution histogram saved to: {save_path}")

def plot_magnitude_stats_by_angle(dataset, angles, save_path):
    """Plot statistical properties of magnitudes grouped by angle"""
    unique_angles = sorted(np.unique(angles))
    
    # Calculate statistics for each angle
    means = []
    medians = []
    stds = []
    maxs = []
    
    for angle in unique_angles:
        angle_indices = np.where(angles == angle)[0]
        angle_specs = dataset.data[angle_indices]
        
        means.append(np.mean(angle_specs))
        medians.append(np.median(angle_specs))
        stds.append(np.std(angle_specs))
        maxs.append(np.max(angle_specs))
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Mean magnitude
    axes[0, 0].plot(unique_angles, means, marker='o', linestyle='-', linewidth=2)
    axes[0, 0].set_title('Mean Magnitude by Angle', fontsize=14)
    axes[0, 0].set_xlabel('Angle (degrees)', fontsize=12)
    axes[0, 0].set_ylabel('Mean Magnitude', fontsize=12)
    axes[0, 0].grid(alpha=0.3)
    
    # Plot 2: Median magnitude
    axes[0, 1].plot(unique_angles, medians, marker='s', linestyle='-', linewidth=2, color='green')
    axes[0, 1].set_title('Median Magnitude by Angle', fontsize=14)
    axes[0, 1].set_xlabel('Angle (degrees)', fontsize=12)
    axes[0, 1].set_ylabel('Median Magnitude', fontsize=12)
    axes[0, 1].grid(alpha=0.3)
    
    # Plot 3: Standard Deviation
    axes[1, 0].plot(unique_angles, stds, marker='^', linestyle='-', linewidth=2, color='red')
    axes[1, 0].set_title('Magnitude Standard Deviation by Angle', fontsize=14)
    axes[1, 0].set_xlabel('Angle (degrees)', fontsize=12)
    axes[1, 0].set_ylabel('Standard Deviation', fontsize=12)
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 4: Maximum magnitude
    axes[1, 1].plot(unique_angles, maxs, marker='*', linestyle='-', linewidth=2, color='purple')
    axes[1, 1].set_title('Maximum Magnitude by Angle', fontsize=14)
    axes[1, 1].set_xlabel('Angle (degrees)', fontsize=12)
    axes[1, 1].set_ylabel('Maximum Magnitude', fontsize=12)
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Magnitude statistics by angle saved to: {save_path}")

def plot_frequency_profile(dataset, angles, save_path):
    """Plot frequency profile (average magnitude per frequency bin)"""
    # Average across time dimension and all samples
    frequency_profile = np.mean(dataset.data, axis=(0, 1))
    
    # Calculate frequency profiles by angle
    unique_angles = sorted(np.unique(angles))
    angle_freq_profiles = []
    
    for angle in unique_angles:
        angle_indices = np.where(angles == angle)[0]
        angle_specs = dataset.data[angle_indices]
        # Average across time and samples
        angle_profile = np.mean(angle_specs, axis=(0, 1))
        angle_freq_profiles.append(angle_profile)
    
    # Plot overall frequency profile
    plt.figure(figsize=(14, 10))
    
    # First subplot: Overall frequency profile
    plt.subplot(2, 1, 1)
    plt.plot(frequency_profile, linewidth=2)
    plt.title('Overall Average Magnitude by Frequency Bin', fontsize=16)
    plt.xlabel('Frequency Bin', fontsize=14)
    plt.ylabel('Average Magnitude', fontsize=14)
    plt.grid(alpha=0.3)
    
    # Second subplot: Frequency profiles by angle
    plt.subplot(2, 1, 2)
    for i, angle in enumerate(unique_angles):
        plt.plot(angle_freq_profiles[i], label=f'{angle}°', linewidth=1.5)
    
    plt.title('Average Magnitude by Frequency Bin (Per Angle)', fontsize=16)
    plt.xlabel('Frequency Bin', fontsize=14)
    plt.ylabel('Average Magnitude', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Frequency profile saved to: {save_path}")

def plot_time_profile(dataset, angles, save_path):
    """Plot time profile (average magnitude per time frame)"""
    # Average across frequency dimension and all samples
    time_profile = np.mean(dataset.data, axis=(0, 2))
    
    # Calculate time profiles by angle
    unique_angles = sorted(np.unique(angles))
    angle_time_profiles = []
    
    for angle in unique_angles:
        angle_indices = np.where(angles == angle)[0]
        angle_specs = dataset.data[angle_indices]
        # Average across frequency and samples
        angle_profile = np.mean(angle_specs, axis=(0, 2))
        angle_time_profiles.append(angle_profile)
    
    # Plot overall time profile
    plt.figure(figsize=(14, 10))
    
    # First subplot: Overall time profile
    plt.subplot(2, 1, 1)
    plt.plot(time_profile, linewidth=2)
    plt.title('Overall Average Magnitude by Time Frame', fontsize=16)
    plt.xlabel('Time Frame', fontsize=14)
    plt.ylabel('Average Magnitude', fontsize=14)
    plt.grid(alpha=0.3)
    
    # Second subplot: Time profiles by angle
    plt.subplot(2, 1, 2)
    for i, angle in enumerate(unique_angles):
        plt.plot(angle_time_profiles[i], label=f'{angle}°', linewidth=1.5)
    
    plt.title('Average Magnitude by Time Frame (Per Angle)', fontsize=16)
    plt.xlabel('Time Frame', fontsize=14)
    plt.ylabel('Average Magnitude', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Time profile saved to: {save_path}")

def main():
    print("Starting STFT Magnitude visualization process...")
    
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
    
    # Load dataset
    dataset = SpectrogramDatasetWithMaterial(
        config.DATA_ROOT,
        config.CLASSES,
        config.SEQ_NUMS,
        selected_freq,
        config.MATERIAL
    )
    
    if len(dataset) == 0:
        print("Dataset is empty!")
        return
    
    print(f"Dataset loaded with {len(dataset)} samples")
    print(f"Spectrogram shape: {dataset.data.shape}")
    
    # Extract angles from class labels - using labels attribute instead of targets
    if hasattr(dataset, 'labels'):
        # Use dataset.labels directly
        labels = dataset.labels
        angles = np.array([get_angle_from_class(dataset.classes[label.item()]) for label in labels])
        print(f"Extracted angles from dataset.labels: {len(angles)} items")
    else:
        # Fallback if labels attribute doesn't exist
        angles = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            angle = get_angle_from_class(dataset.classes[label])
            angles.append(angle)
        angles = np.array(angles)
        print(f"Extracted angles by iterating through dataset: {len(angles)} items")
    
    # Create save directory
    plots_dir = os.path.join(config.SAVE_DIR, 'stft_magnitude_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate visualizations
    print("Generating visualization charts...")
    
    # 1. Plot distribution of magnitude values
    hist_path = os.path.join(plots_dir, f'magnitude_distribution_{config.MATERIAL}_{selected_freq}.png')
    plot_magnitude_distribution(dataset, hist_path)
    
    # 2. Plot magnitude statistics by angle
    stats_path = os.path.join(plots_dir, f'magnitude_stats_by_angle_{config.MATERIAL}_{selected_freq}.png')
    plot_magnitude_stats_by_angle(dataset, angles, stats_path)
    
    # 3. Plot frequency profile
    freq_path = os.path.join(plots_dir, f'frequency_profile_{config.MATERIAL}_{selected_freq}.png')
    plot_frequency_profile(dataset, angles, freq_path)
    
    # 4. Plot time profile
    time_path = os.path.join(plots_dir, f'time_profile_{config.MATERIAL}_{selected_freq}.png')
    plot_time_profile(dataset, angles, time_path)
    
    # 5. Plot spectrogram grid
    grid_path = os.path.join(plots_dir, f'spectrogram_grid_{config.MATERIAL}_{selected_freq}.png')
    plot_spectrogram_grid(dataset, angles, config.MATERIAL, selected_freq, grid_path)
    
    # 6. Create a subdirectory for average spectrograms by angle
    avg_specs_dir = os.path.join(plots_dir, f'avg_spectrograms_{config.MATERIAL}_{selected_freq}')
    os.makedirs(avg_specs_dir, exist_ok=True)
    
    # 7. Plot average spectrograms for each angle
    plot_average_spectrograms_by_angle(dataset, angles, avg_specs_dir)
    
    print("\nVisualization complete! Charts saved to:", plots_dir)
    print(f"Processed {len(dataset)} spectrograms across {len(np.unique(angles))} angles")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc() 