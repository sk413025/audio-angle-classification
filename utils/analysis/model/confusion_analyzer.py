"""
Confusion Matrix Analyzer Module

This module provides tools for analyzing model predictions through confusion matrices:
- Model loading and data preparation
- Angle accuracy computation
- Confusion matrix generation and visualization
"""

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
                          ha="center", va="center", 
                          color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    # Set title and labels
    ax.set_title("Confusion Matrix")
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return output_path

def analyze_angle_predictions(model, dataset, device, output_dir):
    """Analyze model predictions for different angle pairs."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute accuracy for each angle pair
    class_accuracies = compute_angle_accuracy(model, dataset, device)
    
    # Determine unique classes from keys
    unique_classes = set()
    for key in class_accuracies.keys():
        class_i, class_j = key.split('_vs_')
        unique_classes.add(int(class_i))
        unique_classes.add(int(class_j))
    
    unique_classes = sorted(list(unique_classes))
    num_classes = len(unique_classes)
    
    # Create confusion-like matrix for angle comparisons
    angle_matrix = np.zeros((num_classes, num_classes))
    
    for i, class_i in enumerate(unique_classes):
        for j, class_j in enumerate(unique_classes):
            if i == j:
                # Diagonal elements (same class comparison) are set to 100%
                angle_matrix[i, j] = 100
            else:
                key = f"{class_i}_vs_{class_j}"
                reverse_key = f"{class_j}_vs_{class_i}"
                
                if key in class_accuracies:
                    angle_matrix[i, j] = class_accuracies[key]
                elif reverse_key in class_accuracies:
                    # If we only have the reverse comparison, use its complement (100 - accuracy)
                    angle_matrix[i, j] = 100 - class_accuracies[reverse_key]
    
    # Plot the angle accuracy matrix
    plt.figure(figsize=(10, 8))
    im = plt.imshow(angle_matrix, cmap='viridis', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Accuracy (%)')
    
    # Label the axes with angle values
    angle_labels = [f"{angle}°" for angle in unique_classes]
    plt.xticks(np.arange(num_classes), angle_labels)
    plt.yticks(np.arange(num_classes), angle_labels)
    
    # Add text annotations
    for i in range(num_classes):
        for j in range(num_classes):
            text = plt.text(j, i, f"{angle_matrix[i, j]:.1f}",
                           ha="center", va="center", 
                           color="white" if angle_matrix[i, j] < 50 else "black",
                           fontsize=9)
    
    plt.title('Angle Comparison Accuracy')
    plt.xlabel('Higher Angle Class')
    plt.ylabel('Lower Angle Class')
    plt.tight_layout()
    
    # Save the plot
    angle_matrix_path = os.path.join(output_dir, 'angle_accuracy_matrix.png')
    plt.savefig(angle_matrix_path, dpi=150)
    plt.close()
    
    # Create bar plot for average accuracy per angle
    avg_accuracy_per_angle = np.zeros(num_classes)
    for i, class_i in enumerate(unique_classes):
        accuracies = []
        for j, class_j in enumerate(unique_classes):
            if i != j:
                accuracies.append(angle_matrix[i, j])
        avg_accuracy_per_angle[i] = np.mean(accuracies)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(angle_labels, avg_accuracy_per_angle, alpha=0.7)
    
    # Add value labels on top of bars
    for bar, acc in zip(bars, avg_accuracy_per_angle):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f'{acc:.1f}%',
            ha='center',
            fontsize=9
        )
    
    plt.title('Average Accuracy Per Angle')
    plt.xlabel('Angle')
    plt.ylabel('Average Accuracy (%)')
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 105)  # Leave some space for labels
    
    # Save the plot
    angle_accuracy_path = os.path.join(output_dir, 'avg_angle_accuracy.png')
    plt.savefig(angle_accuracy_path, dpi=150)
    plt.close()
    
    # Create matrix showing accuracy difference from expected
    # For angles i and j, we expect high accuracy when i and j are far apart
    # and lower accuracy when they are close
    expected_difficulty = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                expected_difficulty[i, j] = 100  # Diagonal is always perfect
            else:
                # Calculate expected difficulty based on angle difference
                angle_diff = abs(unique_classes[i] - unique_classes[j])
                # Simple linear model: larger difference = easier
                max_diff = max([abs(a - b) for a in unique_classes for b in unique_classes])
                ease = angle_diff / max_diff  # normalized to [0, 1]
                # Scale to a reasonable accuracy range (e.g., 60% to 95%)
                expected_difficulty[i, j] = 60 + 35 * ease
    
    # Calculate difference from expected
    accuracy_difference = angle_matrix - expected_difficulty
    
    # Plot the difference matrix
    plt.figure(figsize=(10, 8))
    # Use diverging colormap to show positive/negative differences
    im = plt.imshow(accuracy_difference, cmap='RdBu', 
                    interpolation='nearest', vmin=-20, vmax=20)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Difference from Expected (%)')
    
    # Label the axes with angle values
    plt.xticks(np.arange(num_classes), angle_labels)
    plt.yticks(np.arange(num_classes), angle_labels)
    
    # Add text annotations
    for i in range(num_classes):
        for j in range(num_classes):
            text = plt.text(j, i, f"{accuracy_difference[i, j]:.1f}",
                           ha="center", va="center", 
                           color="white" if abs(accuracy_difference[i, j]) > 10 else "black",
                           fontsize=9)
    
    plt.title('Angle Comparison: Difference from Expected Accuracy')
    plt.xlabel('Higher Angle Class')
    plt.ylabel('Lower Angle Class')
    plt.tight_layout()
    
    # Save the plot
    diff_matrix_path = os.path.join(output_dir, 'accuracy_difference_matrix.png')
    plt.savefig(diff_matrix_path, dpi=150)
    plt.close()
    
    return {
        'angle_matrix': angle_matrix_path,
        'angle_accuracy': angle_accuracy_path,
        'diff_matrix': diff_matrix_path
    }

def analyze_confusion_matrix(model_path, frequency, material='metal', output_dir='results/confusion_matrix', seed=42):
    """Main function to analyze model performance using confusion matrices."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else 
                         "cpu")
    print(f"Using device: {device}")
    
    # Load model
    try:
        model = load_model(model_path)
        model.to(device)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None
    
    # Prepare data loader
    try:
        data_loader = prepare_data_loader(frequency, material, seed)
        print(f"Data loader prepared for frequency {frequency}, material {material}")
    except Exception as e:
        print(f"Failed to prepare data loader: {e}")
        return None
    
    # Create frequency-specific output directory
    frequency_output_dir = os.path.join(output_dir, frequency)
    os.makedirs(frequency_output_dir, exist_ok=True)
    
    # Generate and plot confusion matrix
    print("Generating confusion matrix...")
    cm = generate_conf_matrix(model, data_loader, device)
    cm_path = os.path.join(frequency_output_dir, 'confusion_matrix.png')
    plot_conf_matrix(cm, cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    
    # Analyze angle-specific predictions
    print("Analyzing angle-specific predictions...")
    angle_output_dir = os.path.join(frequency_output_dir, 'angle_analysis')
    angle_results = analyze_angle_predictions(model, data_loader, device, angle_output_dir)
    print(f"Angle analysis results saved to {angle_output_dir}")
    
    return {
        'confusion_matrix': cm_path,
        'angle_analysis': angle_results
    }