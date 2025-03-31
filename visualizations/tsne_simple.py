"""
Simplified t-SNE example that avoids matplotlib array conversion issues
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs, make_circles, make_moons, make_swiss_roll
import time

from angle_classification_deg6.config import SAVE_DIR

def generate_data(dataset_type='blobs', n_samples=1000, random_state=42):
    """Generate synthetic datasets"""
    print(f"Generating {dataset_type} dataset, samples: {n_samples}")
    
    if dataset_type == 'blobs':
        X, y = make_blobs(n_samples=n_samples, n_features=10, centers=5, 
                          cluster_std=1.0, random_state=random_state)
    elif dataset_type == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=0.05, factor=0.5, random_state=random_state)
        noise_dims = np.random.RandomState(random_state).randn(n_samples, 8)
        X = np.hstack([X, noise_dims])
    elif dataset_type == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=random_state)
        noise_dims = np.random.RandomState(random_state).randn(n_samples, 8)
        X = np.hstack([X, noise_dims])
    elif dataset_type == 'swiss_roll':
        X, color = make_swiss_roll(n_samples=n_samples, noise=0.05, random_state=random_state)
        y = np.round(color * 4).astype(int)
    
    print(f"Data shape: {X.shape}, Classes: {len(np.unique(y))}")
    return X, y

def apply_tsne(X, perplexity=30):
    """Apply t-SNE with simple parameters"""
    print(f"Applying t-SNE with perplexity={perplexity}")
    start_time = time.time()
    
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate='auto',
        init='pca',
        random_state=42
    )
    
    X_tsne = tsne.fit_transform(X)
    print(f"t-SNE shape: {X_tsne.shape}")
    print(f"t-SNE completed in {time.time() - start_time:.2f} seconds")
    return X_tsne

def simple_plot(X_tsne, y, title, filename):
    """Create a simple plot with no fancy features"""
    # Make a new clean figure
    plt.figure(figsize=(8, 6))
    plt.clf()
    
    # Get unique classes
    unique_classes = np.unique(y)
    
    # Create a simple scatter plot, one color per class
    for i in unique_classes:
        # Simple indexing to avoid array conversion issues
        indices = np.where(y == i)[0]
        x_values = X_tsne[indices, 0]
        y_values = X_tsne[indices, 1]
        plt.scatter(x_values, y_values, label=f'Class {i}')
    
    # Add basic labels
    plt.title(title)
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    
    # Add a simple legend
    if len(unique_classes) <= 10:  # Only add legend if not too many classes
        plt.legend()
    
    # Save figure - use low dpi and simplest format
    try:
        plt.savefig(filename, dpi=100)
        plt.close()  # Close immediately to free memory
        print(f"Saved figure to {filename}")
    except Exception as e:
        print(f"Error saving figure: {e}")
        plt.close()

def main():
    """Run simplified t-SNE examples"""
    # Ensure directory exists
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Test on different datasets
    datasets = ['blobs', 'circles', 'moons']  # Skip swiss_roll as it has too many classes
    perplexities = [5, 30, 50]
    
    for dataset_type in datasets:
        print(f"\n{'='*50}\nProcessing {dataset_type} dataset\n{'='*50}")
        
        # Generate data
        X, y = generate_data(dataset_type)
        
        # Process with different perplexities
        for perplexity in perplexities:
            # Apply t-SNE
            X_tsne = apply_tsne(X, perplexity=perplexity)
            
            # Create simple visualization
            filename = os.path.join(SAVE_DIR, f"tsne_{dataset_type}_perp{perplexity}.png")
            simple_plot(X_tsne, y, f"t-SNE: {dataset_type.capitalize()} (perplexity={perplexity})", filename)

if __name__ == "__main__":
    main()
