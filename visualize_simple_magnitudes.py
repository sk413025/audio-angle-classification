"""
Simple STFT Magnitude Visualization Tool
- Calculates average magnitude values for each angle
- Plots average magnitude with angle on the x-axis
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

def calculate_average_magnitudes(dataset, angles):
    """Calculate average magnitude for each angle"""
    unique_angles = sorted(np.unique(angles))
    avg_magnitudes = []
    
    for angle in unique_angles:
        # Get indices of all samples with this angle
        angle_indices = np.where(angles == angle)[0]
        
        # Get spectrograms for this angle
        angle_specs = dataset.data[angle_indices]
        
        # Calculate average magnitude (over all dimensions)
        avg_mag = float(torch.mean(angle_specs))
        avg_magnitudes.append(avg_mag)
        
        print(f"Angle {angle}°: Average magnitude = {avg_mag:.6f}")
    
    return unique_angles, avg_magnitudes

def plot_average_magnitudes(angles, magnitudes, material, frequency, save_path):
    """Create plot of average magnitudes by angle"""
    plt.figure(figsize=(12, 8))
    
    plt.plot(angles, magnitudes, marker='o', linestyle='-', linewidth=2, markersize=10)
    
    # Add data points
    for i, (angle, mag) in enumerate(zip(angles, magnitudes)):
        plt.text(angle, mag, f"{mag:.6f}", ha='center', va='bottom', fontsize=9)
    
    plt.title(f'Average STFT Magnitude by Angle ({material}, {frequency})', fontsize=16)
    plt.xlabel('Angle (degrees)', fontsize=14)
    plt.ylabel('Average Magnitude', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Set x-axis to show all angles
    plt.xticks(angles, [f"{angle}°" for angle in angles])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Average magnitude plot saved to: {save_path}")

def main():
    print("Starting simplified STFT Magnitude visualization...")
    
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
    
    # Calculate average magnitudes
    print("\nCalculating average magnitudes by angle...")
    unique_angles, avg_magnitudes = calculate_average_magnitudes(dataset, angles)
    
    # Plot average magnitudes
    plot_path = os.path.join(plots_dir, f'average_magnitude_by_angle_{config.MATERIAL}_{selected_freq}.png')
    plot_average_magnitudes(unique_angles, avg_magnitudes, config.MATERIAL, selected_freq, plot_path)
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc() 