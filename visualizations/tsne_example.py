"""
t-SNE Example: Demonstrate t-SNE dimensionality reduction on synthetic data
Features:
- Generate various synthetic datasets
- Compare original data with t-SNE reduced data
- Show how different parameter settings affect t-SNE results
- Provide visualization reference for understanding t-SNE behavior
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs, make_circles, make_moons, make_swiss_roll
from matplotlib.colors import ListedColormap
import time
import sys

# IMPORT_CONFIG_COMPLETE
# Add the parent directory to the Python path
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from parent directory
from angle_classification_deg6.config import SAVE_DIR

def generate_data(dataset_type='blobs', n_samples=1000, random_state=42):
    """Generate various synthetic datasets"""
    print(f"Generating {dataset_type} dataset, samples: {n_samples}")
    
    if dataset_type == 'blobs':
        # Generate multiple Gaussian clusters
        X, y = make_blobs(
            n_samples=n_samples, n_features=10, centers=5, 
            cluster_std=1.0, random_state=random_state
        )
        
    elif dataset_type == 'circles':
        # Generate concentric circles
        X, y = make_circles(
            n_samples=n_samples, noise=0.05, factor=0.5, 
            random_state=random_state
        )
        # Add extra noise dimensions to simulate high-dimensional data
        noise_dims = np.random.RandomState(random_state).randn(n_samples, 8)
        X = np.hstack([X, noise_dims])
        
    elif dataset_type == 'moons':
        # Generate two half-moon shapes
        X, y = make_moons(
            n_samples=n_samples, noise=0.1, 
            random_state=random_state
        )
        # Add extra noise dimensions
        noise_dims = np.random.RandomState(random_state).randn(n_samples, 8)
        X = np.hstack([X, noise_dims])
        
    elif dataset_type == 'swiss_roll':
        # Generate swiss roll (3D manifold)
        X, color = make_swiss_roll(
            n_samples=n_samples, noise=0.05,
            random_state=random_state
        )
        y = np.round(color * 4).astype(int)  # Convert continuous color to discrete labels
        
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    print(f"Data shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    return X, y

def apply_tsne_with_params(X, perplexity=30, n_iter=1000, learning_rate='auto', init='pca'):
    """Apply t-SNE with timing"""
    print(f"\nApplying t-SNE reduction (perplexity={perplexity}, max_iter={n_iter}, init={init})")
    start_time = time.time()
    
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,  # Changed from n_iter to max_iter to avoid deprecation warning
        learning_rate=learning_rate,
        init=init,
        random_state=42
    )
    
    X_tsne = tsne.fit_transform(X)
    elapsed = time.time() - start_time
    
    print(f"t-SNE transformation completed in {elapsed:.2f} seconds")
    return X_tsne

def visualize_tsne_results(X, y, X_tsne, dataset_type, params, save_path=None):
    """Visualize comparison between original data (if low-dimensional) and t-SNE results"""
    # Increase recursion limit to avoid errors
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(10000)
    
    try:
        # Create a new figure with explicit figsize
        plt.figure(figsize=(15, 6))
        
        # Create nice color mapping - use direct colormap
        n_classes = len(np.unique(y))
        # Convert to simple integer array to avoid issues
        y_plot = y.astype(int)
        
        # LEFT PLOT: Original data visualization (only for 2D or 3D data)
        if X.shape[1] == 2:
            plt.subplot(121)
            for i in range(n_classes):
                mask = y_plot == i
                plt.scatter(X[mask, 0], X[mask, 1], label=f"Class {i}", s=50, alpha=0.8)
            plt.title('Original Data', fontsize=14)
            plt.grid(alpha=0.3)
            plt.legend()
            
        elif X.shape[1] == 3 or dataset_type == 'swiss_roll':
            ax1 = plt.subplot(121, projection='3d')
            # Plot each class separately
            for i in range(n_classes):
                mask = y_plot == i
                if dataset_type == 'swiss_roll' and X.shape[1] > 3:
                    ax1.scatter(X[mask, 0], X[mask, 1], X[mask, 2], label=f"Class {i}", s=50, alpha=0.8)
                else:
                    ax1.scatter(X[mask, 0], X[mask, 1], X[mask, 2], label=f"Class {i}", s=50, alpha=0.8)
            ax1.set_title('Original Data (3D)', fontsize=14)
            ax1.legend()
        else:
            # High-dimensional data, show explanatory text
            plt.subplot(121)
            plt.text(0.5, 0.5, f"Original data is {X.shape[1]}-dimensional\nCannot visualize directly", 
                    ha='center', va='center', fontsize=14)
            plt.axis('off')
        
        # RIGHT PLOT: t-SNE result visualization
        plt.subplot(122)
        # Plot each class separately
        for i in range(n_classes):
            mask = y_plot == i
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=f"Class {i}", s=50, alpha=0.8)
        plt.title('t-SNE Reduced Data', fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend()
        
        # Add parameter explanation as plain text
        param_text = f"Perplexity: {params['perplexity']}\n" \
                    f"Iterations: {params['n_iter']}\n" \
                    f"Init: {params['init']}"
        plt.figtext(0.5, 0.02, param_text, ha='center', fontsize=12,
                  bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        
        plt.suptitle(f't-SNE Example: {dataset_type.capitalize()} Dataset', fontsize=16)
        
        # Use subplots_adjust instead of tight_layout
        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2, wspace=0.3)
        
        # Save figure with minimal options
        if save_path:
            try:
                plt.savefig(save_path, dpi=150)  # Lower DPI to avoid memory issues
                print(f"Figure saved to {save_path}")
            except Exception as e:
                print(f"Error saving figure: {e}")
        
        plt.close()  # Close the figure to free memory
    except Exception as e:
        print(f"Error in visualization: {e}")
    finally:
        # Restore original recursion limit
        sys.setrecursionlimit(old_limit)

def compare_perplexities(X, y, dataset_type, perplexities=[5, 30, 100], save_path=None):
    """Compare effects of different perplexity values"""
    # Increase recursion limit
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(10000)
    
    try:
        import gc
        # Force garbage collection before creating new plots
        gc.collect()
        
        n_plots = len(perplexities)
        # Use a non-interactive backend for saving
        import matplotlib
        orig_backend = matplotlib.get_backend()
        try:
            matplotlib.use('Agg')  # Switch to non-interactive backend
        except:
            print("Could not switch matplotlib backend, continuing with current backend")
        
        # Create figure with explicit dimensions
        fig = plt.figure(figsize=(5*n_plots, 5))
        
        # Convert to simple integer array and validate
        y_plot = np.asarray(y, dtype=int)
        n_classes = len(np.unique(y_plot))
        
        for i, perplexity in enumerate(perplexities):
            print(f"\nTesting perplexity = {perplexity}")
            X_tsne = apply_tsne_with_params(X, perplexity=perplexity)
            
            # Check for NaN or infinite values
            if np.isnan(X_tsne).any() or np.isinf(X_tsne).any():
                print(f"Warning: NaN or Inf values detected in t-SNE output for perplexity={perplexity}")
                # Replace problematic values
                X_tsne = np.nan_to_num(X_tsne, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Create subplot and get axis
            ax = fig.add_subplot(1, n_plots, i+1)
            
            # Create separate scatter plots for each class to avoid array issues
            for c in range(n_classes):
                mask = y_plot == c
                # Create clean arrays for plotting to avoid any conversion issues
                x_values = np.asarray(X_tsne[mask, 0], dtype=float)
                y_values = np.asarray(X_tsne[mask, 1], dtype=float)
                ax.scatter(x_values, y_values, label=f"Class {c}", s=50, alpha=0.8)
            
            ax.set_title(f'Perplexity = {perplexity}', fontsize=14)
            ax.grid(alpha=0.3)
            
            if i == 0:  # Only add legend to first plot
                ax.legend()
        
        fig.suptitle(f't-SNE Perplexity Comparison: {dataset_type.capitalize()} Dataset', fontsize=16)
        fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.2)
        
        # Save using the simplest method possible with explicit format
        if save_path:
            try:
                # Try saving with minimal processing
                plt.savefig(save_path, dpi=100, bbox_inches=None, format='png')
                print(f"Comparison figure saved to {save_path}")
            except Exception as e:
                print(f"Standard save failed: {e}")
                try:
                    # Try a different format
                    alternate_path = save_path.replace('.png', '.jpg')
                    plt.savefig(alternate_path, format='jpg', dpi=100)
                    print(f"Saved as JPG instead: {alternate_path}")
                except Exception as e2:
                    print(f"Could not save figure: {e2}")
                    print("Skipping figure save to prevent further errors")
        
        # Always close the figure to free memory
        plt.close(fig)
        gc.collect()  # Force garbage collection
        
        # Restore original backend
        try:
            matplotlib.use(orig_backend)
        except:
            pass
            
    except Exception as e:
        print(f"Error in perplexity comparison: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original recursion limit
        sys.setrecursionlimit(old_limit)

def main():
    """Main function: Show t-SNE effects on different datasets"""
    # Ensure save directory exists
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Dataset type list
    dataset_types = ['blobs', 'circles', 'moons', 'swiss_roll']
    
    for dataset_type in dataset_types:
        print(f"\n\n{'='*50}")
        print(f"Processing {dataset_type} dataset")
        print(f"{'='*50}")
        
        # Generate data
        X, y = generate_data(dataset_type=dataset_type, n_samples=1000)
        
        # Set t-SNE parameters
        params = {
            'perplexity': 30,
            'n_iter': 1000,
            'learning_rate': 'auto',
            'init': 'pca'
        }
        
        # Apply t-SNE
        X_tsne = apply_tsne_with_params(X, **params)
        
        # Visualize results
        save_path = os.path.join(SAVE_DIR, f"tsne_example_{dataset_type}.png")
        visualize_tsne_results(X, y, X_tsne, dataset_type, params, save_path)
        
        # Compare different perplexities
        compare_save_path = os.path.join(SAVE_DIR, f"tsne_perplexity_compare_{dataset_type}.png")
        compare_perplexities(X, y, dataset_type, [5, 30, 100], compare_save_path)

if __name__ == "__main__":
    main()
