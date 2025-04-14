import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import argparse
from glob import glob
import re

from config import CLASSES, DATA_ROOT, SEQ_NUMS
from datasets import SpectrogramDatasetWithMaterial, RankingPairDataset
from torch.utils.data import random_split, DataLoader
from utils.common_utils import set_seed, worker_init_fn

def load_model(model_path):
    """Load a trained model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        # Load the model file
        checkpoint = torch.load(model_path)
        
        # If it's a state dict, we need to create and load the model
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            from models.resnet_ranker import SimpleCNNAudioRanker as ResNetAudioRanker
            
            # Create model instance
            # We need to determine n_freqs, ideally from the model structure
            # For now use a default value that should work with our data
            n_freqs = 1025  # This should match the number of frequency bins in our spectrograms
            
            model = ResNetAudioRanker(n_freqs=n_freqs)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            return model
        elif hasattr(checkpoint, 'eval'):
            # It's already a model
            checkpoint.eval()
            return checkpoint
        else:
            raise ValueError("Unsupported model format")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")

def prepare_data_loader(frequency, material, seed=42):
    """Prepare a data loader for the specified frequency and material."""
    # Load the dataset
    dataset = SpectrogramDatasetWithMaterial(
        DATA_ROOT,
        CLASSES,
        SEQ_NUMS,
        frequency,
        material
    )
    
    # Split the dataset into train and validation sets
    train_size = int(0.70 * len(dataset))
    val_size = len(dataset) - train_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    # Create ranking dataset for validation
    val_ranking_dataset = RankingPairDataset(val_dataset)
    
    # Create data loader
    val_dataloader = DataLoader(
        val_ranking_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(seed)
    )
    
    return val_dataloader

def compute_angle_accuracy(model, dataset, device):
    """Compute accuracy for each angle class."""
    model.eval()
    
    # In RankingPairDataset, we need to access the underlying dataset
    base_dataset = dataset.dataset.dataset
    
    # Extract sample data
    all_samples = []
    for i in range(len(base_dataset)):
        data, label = base_dataset[i]
        all_samples.append((data, label))
    
    # Group samples by class
    class_samples = {}
    for data, label in all_samples:
        label_idx = label.item()
        if label_idx not in class_samples:
            class_samples[label_idx] = []
        class_samples[label_idx].append(data)
    
    # Compute pairwise comparisons for each class
    class_accuracies = {}
    all_labels = sorted(class_samples.keys())
    
    # For each pair of classes, compare samples
    for i, class_i in enumerate(all_labels):
        for j, class_j in enumerate(all_labels):
            if i == j:
                continue
                
            key = f"{class_i}_vs_{class_j}"
            
            # Skip if we've already computed the reverse comparison
            if f"{class_j}_vs_{class_i}" in class_accuracies:
                continue
                
            correct = 0
            total = 0
            
            # For each sample in class_i and class_j
            for sample_i in class_samples[class_i]:
                for sample_j in class_samples[class_j]:
                    # Prepare input
                    sample_i = sample_i.unsqueeze(0).to(device)
                    sample_j = sample_j.unsqueeze(0).to(device)
                    
                    # Get predictions
                    with torch.no_grad():
                        output_i = model(sample_i)
                        output_j = model(sample_j)
                    
                    # Determine if prediction is correct
                    expected = class_i > class_j
                    predicted = output_i > output_j
                    
                    if expected == predicted.item():
                        correct += 1
                    total += 1
            
            # Calculate accuracy
            accuracy = 100 * correct / total if total > 0 else 0
            class_accuracies[key] = accuracy
    
    return class_accuracies

def generate_conf_matrix(model, data_loader, device):
    """Generate a confusion matrix for the model."""
    model.eval()
    
    # Collect predictions and targets
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data1, data2, targets, label1, label2 in data_loader:
            data1 = data1.to(device)
            data2 = data2.to(device)
            
            # Get predictions
            output1 = model(data1)
            output2 = model(data2)
            
            # Convert to binary predictions
            preds = (output1 > output2).float().cpu().numpy()
            targets = (targets > 0).float().cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(targets)
    
    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    return cm

def plot_conf_matrix(cm, output_path, class_names=['Lower Angle', 'Higher Angle']):
    """Plot a confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the confusion matrix
    im = ax.imshow(cm, cmap='Blues')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Count', rotation=-90, va='bottom')
    
    # Set up axes
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Loop over data dimensions and create text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.text(j, i, cm[i, j],
                       ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
    
    # Add labels and title
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return output_path

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Analyze model performance with confusion matrices')
    
    parser.add_argument('--model-path', type=str, help='Path to the model file')
    parser.add_argument('--output-dir', type=str, default='confusion_matrix_results', help='Directory to save the results')
    parser.add_argument('--frequency', type=str, default='1000hz', help='Frequency data used for training')
    parser.add_argument('--material', type=str, default='plastic', help='Material type used for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    return parser.parse_args()

def analyze_angle_predictions(model, dataset, device, output_dir):
    """Analyze predictions for each angle pair directly from the base dataset."""
    model.eval()
    
    # Get class names from the dataset
    class_names = dataset.classes
    num_classes = len(class_names)
    
    # Create empty confusion matrix
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    
    # Track accuracies between pairs of angles
    pair_accuracies = {}
    
    # Collect all samples by class
    samples_by_class = {}
    for i in range(len(dataset)):
        data, label = dataset[i]
        label_idx = label.item()
        if label_idx not in samples_by_class:
            samples_by_class[label_idx] = []
        samples_by_class[label_idx].append(data)
    
    print(f"Collected samples for {len(samples_by_class)} classes")
    
    # For each pair of classes
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                continue  # Skip same class comparisons
                
            correct = 0
            total = 0
            
            # Compare samples from class i with samples from class j
            for sample_i in samples_by_class[i]:
                for sample_j in samples_by_class[j]:
                    # Prepare inputs
                    sample_i_tensor = sample_i.unsqueeze(0).to(device)
                    sample_j_tensor = sample_j.unsqueeze(0).to(device)
                    
                    # Get predictions
                    with torch.no_grad():
                        score_i = model(sample_i_tensor).item()
                        score_j = model(sample_j_tensor).item()
                    
                    # Determine prediction (i > j is the expected outcome when i > j)
                    prediction = int(score_i > score_j)
                    expected = int(i > j)
                    
                    # Update confusion counts
                    if prediction == expected:
                        correct += 1
                    
                    # Increment total comparisons
                    total += 1
            
            # Record accuracy for this pair
            accuracy = 100.0 * correct / total if total > 0 else 0
            pair_key = f"{class_names[i]}_vs_{class_names[j]}"
            pair_accuracies[pair_key] = accuracy
            
            # Update confusion matrix 
            # The matrix will show how often the model correctly predicts the relationship
            confusion[i, j] = correct
            confusion[j, i] = total - correct
    
    # Plot and save the confusion matrix
    plt.figure(figsize=(12, 10))
    plt.imshow(confusion, cmap='Blues')
    
    # Add labels and colorbar
    plt.colorbar(label='Correct Predictions')
    plt.xlabel('Class B')
    plt.ylabel('Class A (A > B is expected)')
    plt.title('Confusion Matrix: correctly predicted rank relationships')
    
    # Add class labels
    plt.xticks(range(num_classes), class_names, rotation=45)
    plt.yticks(range(num_classes), class_names)
    
    # Add text annotations
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:
                # Calculate percentage
                total = confusion[i, j] + confusion[j, i]
                percentage = 100.0 * confusion[i, j] / total if total > 0 else 0
                
                # Add text with percentage
                plt.text(j, i, f"{confusion[i, j]}\n({percentage:.1f}%)",
                         ha="center", va="center", 
                         color="white" if confusion[i, j] > confusion.max() / 2 else "black")
    
    plt.tight_layout()
    
    # Save confusion matrix
    matrix_path = os.path.join(output_dir, "angle_confusion_matrix.png")
    plt.savefig(matrix_path, dpi=150)
    plt.close()
    
    # Save pair accuracies
    accuracy_path = os.path.join(output_dir, "angle_pair_accuracies.txt")
    with open(accuracy_path, "w") as f:
        f.write("Angle Pair Accuracy Results\n")
        f.write("==========================\n\n")
        
        # Group by first class
        for i in range(num_classes):
            f.write(f"\n{class_names[i]} comparisons:\n")
            for j in range(num_classes):
                if i != j:
                    pair_key = f"{class_names[i]}_vs_{class_names[j]}"
                    f.write(f"  {pair_key}: {pair_accuracies.get(pair_key, 0):.2f}%\n")
    
    return {
        'confusion_matrix': matrix_path,
        'accuracies': accuracy_path,
        'pair_accuracies': pair_accuracies
    }

def main():
    """Main function."""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Determine device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model_path = args.model_path
    if not model_path:
        # Find most recent model file
        model_dirs = glob(f"/Users/sbplab/Hank/angle_classification_deg6/saved_models/model_checkpoints/plastic_{args.frequency}_*_*")
        if not model_dirs:
            print(f"No model directories found for frequency {args.frequency}")
            return
        
        latest_dir = sorted(model_dirs, key=lambda x: re.search(r'(\d{8}_\d{6})', x).group(1) if re.search(r'(\d{8}_\d{6})', x) else '')[-1]
        model_files = glob(os.path.join(latest_dir, "model_epoch_*.pt"))
        if not model_files:
            print(f"No model files found in {latest_dir}")
            return
        
        # Get model from last epoch
        model_path = sorted(model_files, key=lambda x: int(re.search(r'model_epoch_(\d+).pt', x).group(1)))[-1]
    
    print(f"Loading model from {model_path}")
    model = load_model(model_path)
    model.to(device)
    
    # Load dataset directly (not as ranking pairs)
    print(f"Preparing dataset for {args.frequency}, {args.material}")
    dataset = SpectrogramDatasetWithMaterial(
        DATA_ROOT,
        CLASSES,
        SEQ_NUMS,
        args.frequency,
        args.material
    )
    
    # Analyze angle predictions
    print("Analyzing angle predictions")
    results_dir = os.path.join(args.output_dir, f"{args.frequency}_{os.path.basename(model_path).replace('.pt', '')}")
    os.makedirs(results_dir, exist_ok=True)
    
    results = analyze_angle_predictions(model, dataset, device, results_dir)
    
    print(f"Angle confusion matrix saved to {results['confusion_matrix']}")
    print(f"Angle pair accuracies saved to {results['accuracies']}")

if __name__ == "__main__":
    main() 