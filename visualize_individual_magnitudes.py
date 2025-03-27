"""
Individual STFT Magnitude Visualization Tool
- Plots magnitude values for each individual data point
- Shows distribution of magnitude values across angles
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add current directory to path to ensure modules can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try different import approaches
try:
    # Method 1: Import from angle_classification_deg6 subdirectory
    from angle_classification_deg6.datasets import SpectrogramDatasetWithMaterial
    from angle_classification_deg6 import config
    print("Successfully imported modules from angle classification subdirectory")
except ImportError:
    try:
        # Method 2: Assume files are in current directory
        from datasets import SpectrogramDatasetWithMaterial
        import config
        print("Successfully imported modules from current directory")
    except ImportError:
        # Method 3: Try importing from parent directory
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            from datasets import SpectrogramDatasetWithMaterial
            import config
            print("Successfully imported modules from parent directory")
        except ImportError:
            print("Could not find required modules. Please check file paths.")
            print("Please place datasets.py and config.py in the same directory as this script, or adjust the paths.")
            sys.exit(1)

def get_angle_from_class(class_name):
    """Extract angle value from class name"""
    # Assuming class name format is 'degXXX', where XXX is the angle value
    try:
        return int(class_name[3:])  # Extract number after 'deg'
    except ValueError:
        return 0

def calculate_individual_magnitudes(dataset, angles):
    """Calculate magnitude for each individual data point"""
    # List to store results
    individual_magnitudes = []
    
    # For each sample in the dataset
    for i in range(len(dataset.data)):
        # Get the spectrogram
        spectrogram = dataset.data[i]
        
        # Calculate average magnitude for this sample
        magnitude = float(torch.mean(spectrogram))
        
        # Store angle and magnitude
        individual_magnitudes.append((angles[i], magnitude))
        
        print(f"Sample {i+1}: Angle {angles[i]}°, Magnitude = {magnitude:.6f}")
    
    return individual_magnitudes

def plot_individual_magnitudes(magnitudes, material, frequency, save_path):
    """Create scatter plot of magnitudes by angle for each data point"""
    plt.figure(figsize=(14, 10))
    
    # Extract angles and magnitude values
    angles = [item[0] for item in magnitudes]
    mags = [item[1] for item in magnitudes]
    
    # Get unique angles for plotting
    unique_angles = sorted(np.unique(angles))
    
    # Create scatter plot
    plt.scatter(angles, mags, alpha=0.7, s=100, c='blue', edgecolors='black')
    
    # Add a small amount of jitter to x-values for better visibility
    for angle, mag in magnitudes:
        jitter = np.random.uniform(-2, 2)  # small random offset
        plt.scatter(angle + jitter, mag, alpha=0.7, s=100, c='blue', edgecolors='black')
    
    # Calculate and plot mean values per angle
    for angle in unique_angles:
        # Get all magnitudes for this angle
        angle_mags = [mag for ang, mag in magnitudes if ang == angle]
        mean_mag = np.mean(angle_mags)
        
        # Plot mean with red diamond
        plt.scatter(angle, mean_mag, color='red', marker='D', s=150, 
                    label=f"Mean ({angle}°)" if angle == unique_angles[0] else "")
        
        # Annotate with the mean value
        plt.annotate(f"{mean_mag:.6f}", (angle, mean_mag), 
                     xytext=(0, 10), textcoords='offset points',
                     ha='center', va='bottom', color='red', fontweight='bold')
    
    plt.title(f'Individual STFT Magnitudes by Angle ({material}, {frequency})', fontsize=16)
    plt.xlabel('Angle (degrees)', fontsize=14)
    plt.ylabel('Magnitude', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Set x-axis to show all angles
    plt.xticks(unique_angles, [f"{angle}°" for angle in unique_angles])
    
    # Create legend
    plt.legend(['Individual samples', 'Mean value'])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Individual magnitude plot saved to: {save_path}")

def plot_boxplot_magnitudes(magnitudes, material, frequency, save_path):
    """Create box plot showing magnitude distribution by angle"""
    plt.figure(figsize=(14, 10))
    
    # Group magnitudes by angle
    angles = sorted(set([item[0] for item in magnitudes]))
    angle_groups = {angle: [] for angle in angles}
    
    for angle, mag in magnitudes:
        angle_groups[angle].append(mag)
    
    # Create boxplot
    boxplot_data = [angle_groups[angle] for angle in angles]
    plt.boxplot(boxplot_data, labels=[f"{angle}°" for angle in angles], patch_artist=True)
    
    # Add individual points
    for i, angle in enumerate(angles):
        # Get x position (1-based indexing for boxplot positions)
        x = i + 1
        
        # Get magnitudes for this angle and add jitter
        ys = angle_groups[angle]
        xs = np.random.normal(x, 0.05, size=len(ys))
        
        # Plot individual points
        plt.scatter(xs, ys, alpha=0.6, s=50, c='blue', edgecolors='black')
    
    plt.title(f'Magnitude Distribution by Angle ({material}, {frequency})', fontsize=16)
    plt.xlabel('Angle (degrees)', fontsize=14)
    plt.ylabel('Magnitude', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Magnitude distribution boxplot saved to: {save_path}")

def main():
    print("Starting individual STFT Magnitude visualization...")
    
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
    
    # Extract angles from class labels
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
    
    # Calculate individual magnitudes
    print("\nCalculating magnitudes for each data point...")
    individual_magnitudes = calculate_individual_magnitudes(dataset, angles)
    
    # Plot individual magnitudes (scatter plot)
    scatter_path = os.path.join(plots_dir, f'individual_magnitudes_{config.MATERIAL}_{selected_freq}.png')
    plot_individual_magnitudes(individual_magnitudes, config.MATERIAL, selected_freq, scatter_path)
    
    # Plot magnitude distributions (box plot)
    box_path = os.path.join(plots_dir, f'magnitude_boxplot_{config.MATERIAL}_{selected_freq}.png')
    plot_boxplot_magnitudes(individual_magnitudes, config.MATERIAL, selected_freq, box_path)
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc() 