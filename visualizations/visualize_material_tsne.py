"""
t-SNE Visualization for SpectrogramDatasetWithMaterial
- Visualizes train and test data together using t-SNE
- Colors points by dataset (train/test) and material class
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from sklearn.manifold import TSNE
import time
import sys
# IMPORT_CONFIG_COMPLETE
# Add the parent directory to the Python path
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import gc
import torch
from torch.utils.data import DataLoader

# Add the project root directory to the path to help find modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Try to load the dataset class directly from specific files


from angle_classification_deg6.config import SAVE_DIR

def extract_features_from_dataloader(dataloader):
    """Extract features and labels from a dataloader"""
    features = []
    labels = []
    
    print("Extracting features from dataloader...")
    for batch in dataloader:
        data, label = batch
        
        if isinstance(data, torch.Tensor):
            data = data.numpy()
            
        if len(data.shape) > 2:
            batch_size = data.shape[0]
            data = data.reshape(batch_size, -1)
        
        if isinstance(label, torch.Tensor):
            label = label.numpy()
            
        features.append(data)
        labels.append(label)
    
    features = np.vstack(features)
    labels = np.concatenate(labels)
    
    print(f"Extracted features shape: {features.shape}, labels shape: {labels.shape}")
    return features, labels

def apply_tsne_to_combined(X_train, X_test, perplexity=30, n_iter=1000):
    """Apply t-SNE to combined train and test data"""
    print(f"Applying t-SNE with perplexity={perplexity}, max_iter={n_iter}")
    
    X_combined = np.vstack([X_train, X_test])
    X_combined = np.nan_to_num(X_combined, nan=0.0, posinf=0.0, neginf=0.0)
    
    start_time = time.time()
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        learning_rate='auto',
        init='pca',
        random_state=42
    )
    
    X_tsne = tsne.fit_transform(X_combined)
    elapsed = time.time() - start_time
    print(f"t-SNE completed in {elapsed:.2f} seconds")
    
    X_train_tsne = X_tsne[:len(X_train)]
    X_test_tsne = X_tsne[len(X_train):]
    
    return X_train_tsne, X_test_tsne

def visualize_combined_tsne(X_train_tsne, y_train, X_test_tsne, y_test, save_path=None):
    """Visualize t-SNE results with both train and test data"""
    gc.collect()
    
    try:
        fig = matplotlib.figure.Figure(figsize=(12, 10))
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        
        all_labels = np.unique(np.concatenate([y_train, y_test]))
        n_classes = len(all_labels)
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        
        for i, label in enumerate(all_labels):
            mask = y_train == label
            if np.any(mask):
                ax.scatter(
                    X_train_tsne[mask, 0], 
                    X_train_tsne[mask, 1],
                    color=colors[i],
                    marker='o',
                    s=40, 
                    alpha=0.7,
                    label=f"Train - Material {label}"
                )
        
        for i, label in enumerate(all_labels):
            mask = y_test == label
            if np.any(mask):
                ax.scatter(
                    X_test_tsne[mask, 0], 
                    X_test_tsne[mask, 1],
                    color=colors[i],
                    marker='^',
                    s=40, 
                    alpha=0.7,
                    label=f"Test - Material {label}"
                )
        
        ax.set_title("t-SNE of SpectrogramDatasetWithMaterial: Train vs Test", fontsize=16)
        ax.grid(alpha=0.3)
        
        legend = ax.legend(loc='upper right', fontsize=10)
        
        if save_path:
            canvas.print_figure(save_path, dpi=150)
            print(f"Combined visualization saved to {save_path}")
        
        fig.clear()
        gc.collect()
        
    except Exception as e:
        print(f"Error in visualization: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to visualize SpectrogramDatasetWithMaterial using t-SNE"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    try:
        print("Loading datasets...")
        
        # Create train and test datasets
        from angle_classification_deg6.config import DATA_ROOT, CLASSES, SEQ_NUMS, MATERIAL
        
        # Get training and testing sequences
        train_seqs = SEQ_NUMS[:7]  # first 7 sequences for training
        test_seqs = SEQ_NUMS[7:]   # remaining for testing
        
        # Initialize datasets with the same parameters as in main.py
        train_dataset = SpectrogramDatasetWithMaterial(
            data_dir=DATA_ROOT,
            classes=CLASSES,
            selected_seqs=train_seqs,
            selected_freq='1000hz',
            material=MATERIAL
        )
        
        test_dataset = SpectrogramDatasetWithMaterial(
            data_dir=DATA_ROOT,
            classes=CLASSES,
            selected_seqs=test_seqs,
            selected_freq='1000hz',
            material=MATERIAL
        )
        
        # Create dataloaders
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Extract features
        X_train, y_train = extract_features_from_dataloader(train_loader)
        X_test, y_test = extract_features_from_dataloader(test_loader)
        
        print(f"Train data: {X_train.shape}, Test data: {X_test.shape}")
        
        # Apply t-SNE to both datasets
        X_train_tsne, X_test_tsne = apply_tsne_to_combined(
            X_train, X_test, perplexity=30, n_iter=1000
        )
        
        # Visualize combined results
        save_path = os.path.join(SAVE_DIR, "material_train_test_tsne.png")
        visualize_combined_tsne(X_train_tsne, y_train, X_test_tsne, y_test, save_path)
        
        print("Visualization complete!")
        
    except Exception as e:
        print(f"Error in main process: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
