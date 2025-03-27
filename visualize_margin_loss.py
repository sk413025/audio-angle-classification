"""
Margin Ranking Loss Visualization Tool
Features:
- Load trained models
- Calculate margin ranking loss for each pair in the dataset
- Visualize loss distribution in multiple ways
- Analyze loss characteristics across different angles and frequencies
- Include t-SNE visualization of loss patterns
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # Set backend
from matplotlib.backends.backend_agg import FigureCanvasAgg
from sklearn.manifold import TSNE
import sys
import time

# Add current directory to path to ensure modules can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try different import approaches
try:
    # Method 1: Import from angle_classification_deg6 subdirectory
    from angle_classification_deg6.datasets import SpectrogramDatasetWithMaterial, RankingPairDataset
    from angle_classification_deg6.simple_cnn_models import SimpleCNNAudioRanker
    from angle_classification_deg6 import config
    print("Successfully imported modules from angle classification subdirectory")
except ImportError:
    try:
        # Method 2: Assume files are in current directory
        from datasets import SpectrogramDatasetWithMaterial, RankingPairDataset
        from simple_cnn_models import SimpleCNNAudioRanker
        import config
        print("Successfully imported modules from current directory")
    except ImportError:
        # Method 3: Try importing from parent directory
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            from datasets import SpectrogramDatasetWithMaterial, RankingPairDataset
            from simple_cnn_models import SimpleCNNAudioRanker
            import config
            print("Successfully imported modules from parent directory")
        except ImportError:
            print("Could not find required modules. Please check file paths.")
            print("Please place datasets.py, simple_cnn_models.py, and config.py in the same directory as this script, or adjust the paths.")
            sys.exit(1)

def get_angle_from_class(class_name):
    """Extract angle value from class name"""
    # Assuming class name format is 'degXXX', where XXX is the angle value
    try:
        return int(class_name[3:])  # Extract number after 'deg'
    except ValueError:
        return 0

def calculate_margin_loss(model, dataloader, device, margin=config.MARGIN):
    """
    Calculate margin ranking loss for each pair in the dataset
    Returns: loss values, corresponding angle pairs, labels and predicted scores
    """
    model.eval()
    criterion = nn.MarginRankingLoss(margin=margin, reduction='none')  # Set to not average
    
    all_losses = []
    all_angle_pairs = []
    all_targets = []
    all_scores1 = []
    all_scores2 = []
    all_features = []
    
    with torch.no_grad():
        for data1, data2, targets, label1, label2 in dataloader:
            # Move data to specified device
            data1, data2 = data1.to(device), data2.to(device)
            targets = targets.to(device)
            
            # Get model outputs
            outputs1 = model(data1).view(-1)
            outputs2 = model(data2).view(-1)
            
            # Calculate loss
            loss_values = criterion(outputs1, outputs2, targets)
            
            # Get angle values from class labels
            angles1 = [get_angle_from_class(dataloader.dataset.dataset.classes[label.item()]) 
                       for label in label1]
            angles2 = [get_angle_from_class(dataloader.dataset.dataset.classes[label.item()]) 
                       for label in label2]
            
            # Save results
            all_losses.extend(loss_values.cpu().numpy())
            all_angle_pairs.extend([(a1, a2) for a1, a2 in zip(angles1, angles2)])
            all_targets.extend(targets.cpu().numpy())
            all_scores1.extend(outputs1.cpu().numpy())
            all_scores2.extend(outputs2.cpu().numpy())
            
            # Create a feature vector for each pair that includes angles and scores
            for a1, a2, s1, s2, t in zip(angles1, angles2, 
                                        outputs1.cpu().numpy(), 
                                        outputs2.cpu().numpy(),
                                        targets.cpu().numpy()):
                all_features.append([a1, a2, s1, s2, t])
    
    return all_losses, all_angle_pairs, all_targets, all_scores1, all_scores2, np.array(all_features)

def plot_loss_histogram(losses, save_path):
    """Plot histogram of loss values"""
    plt.figure(figsize=(10, 6))
    
    # Use seaborn to enhance chart
    sns.histplot(losses, bins=30, kde=True)
    
    plt.title('Margin Ranking Loss Distribution', fontsize=15)
    plt.xlabel('Loss Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(alpha=0.3)
    
    # Add mean and median markers
    mean_loss = np.mean(losses)
    median_loss = np.median(losses)
    plt.axvline(mean_loss, color='r', linestyle='--', label=f'Mean: {mean_loss:.4f}')
    plt.axvline(median_loss, color='g', linestyle='-.', label=f'Median: {median_loss:.4f}')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Loss histogram saved to: {save_path}")

def plot_loss_by_angle_diff(losses, angle_pairs, targets, save_path):
    """Plot relationship between loss values and angle differences"""
    plt.figure(figsize=(12, 7))
    
    # Calculate angle differences
    angle_diffs = [abs(a2 - a1) for a1, a2 in angle_pairs]
    
    # Create scatter plot (positive and negative examples separately)
    positive_mask = np.array(targets) > 0
    negative_mask = np.array(targets) < 0
    
    plt.scatter(np.array(angle_diffs)[positive_mask], 
                np.array(losses)[positive_mask], 
                alpha=0.6, c='blue', label='Positive (y=1)')
    
    plt.scatter(np.array(angle_diffs)[negative_mask], 
                np.array(losses)[negative_mask], 
                alpha=0.6, c='red', label='Negative (y=-1)')
    
    # Add trend line
    plt.plot(np.unique(angle_diffs), 
             np.poly1d(np.polyfit(angle_diffs, losses, 1))(np.unique(angle_diffs)), 
             color='black', linestyle='--')
    
    plt.title('Margin Ranking Loss vs Angle Difference', fontsize=15)
    plt.xlabel('Angle Difference (degrees)', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Angle difference relationship chart saved to: {save_path}")

def plot_loss_heatmap(losses, angle_pairs, save_path):
    """Plot heatmap of losses between angle pairs"""
    # Get all possible angles
    all_angles = sorted(list(set([a for pair in angle_pairs for a in pair])))
    n_angles = len(all_angles)
    
    # Create angle index mapping
    angle_to_idx = {angle: i for i, angle in enumerate(all_angles)}
    
    # Initialize heatmap matrix
    heatmap_matrix = np.zeros((n_angles, n_angles))
    count_matrix = np.zeros((n_angles, n_angles))
    
    # Fill heatmap data
    for (a1, a2), loss in zip(angle_pairs, losses):
        i, j = angle_to_idx[a1], angle_to_idx[a2]
        heatmap_matrix[i, j] += loss
        count_matrix[i, j] += 1
    
    # Calculate average loss
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_heatmap = np.divide(heatmap_matrix, count_matrix)
        avg_heatmap = np.nan_to_num(avg_heatmap)
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(avg_heatmap, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=all_angles, yticklabels=all_angles)
    
    plt.title('Average Margin Ranking Loss Between Angle Pairs', fontsize=15)
    plt.xlabel('Angle 2', fontsize=12)
    plt.ylabel('Angle 1', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Loss heatmap saved to: {save_path}")

def plot_score_distribution(scores1, scores2, targets, save_path):
    """Plot distribution of model scores"""
    plt.figure(figsize=(12, 7))
    
    # Convert labels to categories
    target_labels = ["Negative (y=-1)" if t < 0 else "Positive (y=1)" for t in targets]
    
    # Create data frame
    data = {
        "Score1": scores1,
        "Score2": scores2,
        "Label": target_labels
    }
    
    # Create scatter plot
    color_map = {"Positive (y=1)": "blue", "Negative (y=-1)": "red"}
    for label, color in color_map.items():
        mask = np.array(target_labels) == label
        plt.scatter(np.array(scores1)[mask], np.array(scores2)[mask],
                    alpha=0.6, c=color, label=label)
    
    # Add diagonal line
    min_score = min(min(scores1), min(scores2))
    max_score = max(max(scores1), max(scores2))
    plt.plot([min_score, max_score], [min_score, max_score], 
             'k--', alpha=0.5, label='Equal line')
    
    plt.title('Model Score Distribution', fontsize=15)
    plt.xlabel('Score 1', fontsize=12)
    plt.ylabel('Score 2', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Score distribution chart saved to: {save_path}")

def apply_tsne(features, perplexity=30, n_iter=1000):
    """Apply t-SNE dimension reduction with timing"""
    print(f"\nApplying t-SNE reduction (perplexity={perplexity}, max_iter={n_iter})")
    start_time = time.time()
    
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(features) - 1),  # Ensure perplexity is less than sample count
        n_iter=n_iter,
        random_state=42,
        verbose=1
    )
    
    features_tsne = tsne.fit_transform(features)
    elapsed = time.time() - start_time
    
    print(f"t-SNE transformation completed in {elapsed:.2f} seconds")
    return features_tsne

def plot_tsne_visualization(features, losses, targets, save_path):
    """Create t-SNE visualization of features colored by loss value"""
    # Apply t-SNE to feature vectors
    features_tsne = apply_tsne(features)
    
    # Create main plot
    plt.figure(figsize=(15, 12))
    
    # First subplot: colored by loss value
    plt.subplot(2, 1, 1)
    scatter = plt.scatter(
        features_tsne[:, 0],
        features_tsne[:, 1],
        c=losses,
        cmap='viridis',
        s=80,
        alpha=0.7
    )
    
    plt.colorbar(scatter, label='Loss Value')
    plt.title('t-SNE Visualization of Margin Ranking Loss', fontsize=16)
    plt.xlabel('t-SNE dimension 1', fontsize=12)
    plt.ylabel('t-SNE dimension 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Second subplot: colored by target
    plt.subplot(2, 1, 2)
    colors = ['red' if t < 0 else 'blue' for t in targets]
    plt.scatter(
        features_tsne[:, 0],
        features_tsne[:, 1],
        c=colors,
        s=80,
        alpha=0.7
    )
    
    # Add legend manually
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
               markersize=10, label='Positive (y=1)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
               markersize=10, label='Negative (y=-1)')
    ]
    plt.legend(handles=legend_elements)
    
    plt.title('t-SNE Visualization by Target Label', fontsize=16)
    plt.xlabel('t-SNE dimension 1', fontsize=12)
    plt.ylabel('t-SNE dimension 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"t-SNE visualization saved to: {save_path}")

def main():
    print("Starting Margin Ranking Loss visualization process...")
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
        if f.startswith('simple_cnn_') 
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
        material = parts[2]
        frequency = parts[3]
    except IndexError:
        print("Error parsing model filename. Using default values.")
        material = "unknown"
        frequency = "unknown"
    
    # Load dataset
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
    
    # Create ranking pair dataset
    ranking_dataset = RankingPairDataset(dataset)
    
    # Set batch size
    batch_size = min(32, len(ranking_dataset))
    
    # Create data loader
    dataloader = DataLoader(
        ranking_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Created data loader with {len(dataloader)} batches")
    
    # Load model
    model = SimpleCNNAudioRanker(n_freqs=dataset.data.shape[2])
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print("Calculating margin ranking loss...")
    losses, angle_pairs, targets, scores1, scores2, features = calculate_margin_loss(
        model, dataloader, device
    )
    
    # Create save directory
    plots_dir = os.path.join(config.SAVE_DIR, 'margin_loss_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate visualizations
    print("Generating visualization charts...")
    
    # 1. Loss histogram
    hist_path = os.path.join(plots_dir, f'loss_histogram_{material}_{frequency}.png')
    plot_loss_histogram(losses, hist_path)
    
    # 2. Angle difference relationship chart
    angle_path = os.path.join(plots_dir, f'loss_by_angle_diff_{material}_{frequency}.png')
    plot_loss_by_angle_diff(losses, angle_pairs, targets, angle_path)
    
    # 3. Loss heatmap
    heatmap_path = os.path.join(plots_dir, f'loss_heatmap_{material}_{frequency}.png')
    plot_loss_heatmap(losses, angle_pairs, heatmap_path)
    
    # 4. Score distribution chart
    score_path = os.path.join(plots_dir, f'score_distribution_{material}_{frequency}.png')
    plot_score_distribution(scores1, scores2, targets, score_path)
    
    # 5. t-SNE visualization
    tsne_path = os.path.join(plots_dir, f'tsne_visualization_{material}_{frequency}.png')
    plot_tsne_visualization(features, losses, targets, tsne_path)
    
    print("\nVisualization complete! Charts saved to:", plots_dir)
    print(f"Processed {len(losses)} ranking pairs")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc() 