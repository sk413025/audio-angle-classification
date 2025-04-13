"""
Common plotting utilities for visualization across the project.

This module provides shared functions for plot styling, configuration,
and other general plotting tasks used by multiple visualization components.
"""

import matplotlib.pyplot as plt
import os

def set_plot_style(style='default'):
    """Set the global matplotlib style for consistent visualization.
    
    Args:
        style (str): The style to use. Options include 'default', 'scientific', etc.
                     Default is 'default'.
    """
    if style == 'scientific':
        plt.style.use('seaborn-v0_8-whitegrid')
    elif style == 'presentation':
        plt.style.use('seaborn-v0_8-talk')
    else:  # default
        # Use more generic style that should be widely available
        plt.style.use('ggplot')

def ensure_directory_exists(directory):
    """Ensure that the target directory exists, creating it if necessary.
    
    Args:
        directory (str): Path to the directory to check/create.
    """
    os.makedirs(directory, exist_ok=True)
    
def save_figure(fig, save_path, dpi=300, bbox_inches='tight'):
    """Save a matplotlib figure with standardized settings.
    
    Args:
        fig (matplotlib.figure.Figure): The figure to save
        save_path (str): Path where the figure should be saved
        dpi (int): Resolution in dots per inch
        bbox_inches (str): Bounding box setting
    """
    ensure_directory_exists(os.path.dirname(save_path))
    fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)  # Close figure to free memory
    
def create_figure(figsize=(12, 10)):
    """Create a figure with standard sizing.
    
    Args:
        figsize (tuple): Width, height in inches
        
    Returns:
        matplotlib.figure.Figure: A new figure
    """
    return plt.figure(figsize=figsize) 