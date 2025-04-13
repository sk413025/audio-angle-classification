"""
Training visualization utilities for the audio angle classification project.

This module provides functions for visualizing training metrics such as
loss and accuracy over epochs.
"""

import matplotlib.pyplot as plt
import os
from utils.visualization.plot_utils import create_figure, save_figure, ensure_directory_exists

def plot_training_history(
    history, 
    save_path, 
    title_prefix="Training and Validation", 
    figsize=(12, 10),
    include_standard_loss=True
):
    """Plot training history metrics including loss and accuracy.
    
    Args:
        history (dict): Dictionary containing training history with keys:
            - 'epoch' (list): Epoch numbers
            - 'train_loss_main' (list): Training loss values 
            - 'val_loss_main' (list): Validation loss values
            - 'train_accuracy' (list): Training accuracy values
            - 'val_accuracy' (list): Validation accuracy values
            - 'train_loss_standard_log' (list, optional): Standard loss values for training
            - 'val_loss_standard_log' (list, optional): Standard loss values for validation
            - 'args' (dict): Arguments used for training including loss_type, frequency, material
            
        save_path (str): Path where the plot will be saved
        title_prefix (str): Prefix for plot titles
        figsize (tuple): Figure size (width, height) in inches
        include_standard_loss (bool): Whether to include standard loss plots for GHM loss
        
    Returns:
        str: Path where the plot was saved
    """
    # Extract training parameters
    args = history.get('args', {})
    loss_type = args.get('loss_type', 'unknown')
    frequency = args.get('frequency', 'unknown')
    material = args.get('material', 'unknown')
    
    # Create figure with two subplots
    fig = create_figure(figsize)
    
    # Loss plot
    plt.subplot(2, 1, 1)
    plt.plot(history['epoch'], history['train_loss_main'], 'b-', label=f'Train Loss ({loss_type})')
    plt.plot(history['epoch'], history['val_loss_main'], 'r-', label=f'Val Loss ({loss_type})')
    
    # Optionally plot standard loss if GHM was used and include_standard_loss is True
    if include_standard_loss and loss_type == 'ghm' and 'train_loss_standard_log' in history and 'val_loss_standard_log' in history:
        plt.plot(history['epoch'], history['train_loss_standard_log'], 'c--', label='Train Loss (Standard Log)')
        plt.plot(history['epoch'], history['val_loss_standard_log'], 'm--', label='Val Loss (Standard Log)')
    
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{title_prefix} Loss ({frequency} / {material})')
    
    # Accuracy plot
    plt.subplot(2, 1, 2)
    plt.plot(history['epoch'], history['train_accuracy'], 'b-', label='Train Accuracy')
    plt.plot(history['epoch'], history['val_accuracy'], 'r-', label='Val Accuracy')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title(f'{title_prefix} Accuracy ({frequency} / {material})')
    
    plt.tight_layout()
    
    # Save figure
    ensure_directory_exists(os.path.dirname(save_path))
    save_figure(fig, save_path)
    
    return save_path 