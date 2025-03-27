"""
t-SNE Visualization with compatibility fixes
- Provides visualization utilities for t-SNE results
- Uses Matplotlib backend switching to avoid version compatibility issues
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 設置後端
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg  # 正確的導入方式
from sklearn.manifold import TSNE
import time
import sys
from mpl_toolkits.mplot3d import Axes3D
import gc
import torch
import seaborn as sns
import config  # 導入 config
import torch.nn as nn

from config import SAVE_DIR
from datasets import SpectrogramDatasetWithMaterial
from simple_cnn_models import SimpleCNNAudioRanker

def apply_tsne_with_params(X, perplexity=30, n_iter=1000, learning_rate='auto', init='pca'):
    """Apply t-SNE with timing"""
    print(f"\nApplying t-SNE reduction (perplexity={perplexity}, max_iter={n_iter}, init={init})")
    start_time = time.time()
    
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        learning_rate=learning_rate,
        init=init,
        random_state=42
    )
    
    X_tsne = tsne.fit_transform(X)
    elapsed = time.time() - start_time
    
    print(f"t-SNE transformation completed in {elapsed:.2f} seconds")
    return X_tsne

def visualize_combined_tsne(X_list, y_list, titles, perplexity=30, save_path=None):
    """
    Visualize multiple t-SNE results side by side
    Compatible with older matplotlib versions
    """
    # Force garbage collection before creating new plots
    gc.collect()
    
    try:
        n_plots = len(X_list)
        # Create figure with basic parameters
        # Use Figure object directly instead of pyplot function
        fig = matplotlib.figure.Figure(figsize=(12, 10))
        
        # Create a custom canvas if needed
        canvas = matplotlib.backends.backend_agg.FigureCanvasAgg(fig)
        
        for i, (X, y, title) in enumerate(zip(X_list, y_list, titles)):
            # Apply t-SNE
            X_tsne = apply_tsne_with_params(X, perplexity=perplexity)
            
            # Check for NaN or infinite values
            if np.isnan(X_tsne).any() or np.isinf(X_tsne).any():
                print(f"Warning: NaN or Inf values detected in t-SNE output for {title}")
                X_tsne = np.nan_to_num(X_tsne, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Get subplot axes
            ax = fig.add_subplot(1, n_plots, i+1)
            
            # Convert to integer array
            y_plot = np.asarray(y, dtype=int)
            n_classes = len(np.unique(y_plot))
            
            # Create separate scatter plots for each class
            for c in range(n_classes):
                mask = y_plot == c
                x_values = np.asarray(X_tsne[mask, 0], dtype=float)
                y_values = np.asarray(X_tsne[mask, 1], dtype=float)
                ax.scatter(x_values, y_values, label=f"Class {c}", s=50, alpha=0.8)
            
            ax.set_title(f'{title}', fontsize=14)
            ax.grid(alpha=0.3)
            ax.legend()
        
        fig.suptitle('t-SNE Comparisons', fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout
        
        # Save figure (with error handling)
        if save_path:
            try:
                canvas.print_figure(save_path, dpi=100)
                print(f"Combined figure saved to {save_path}")
            except Exception as e:
                print(f"Error saving combined figure: {e}")
                try:
                    # Try alternate format
                    alt_path = save_path.replace('.png', '.pdf')
                    canvas.print_figure(alt_path, format='pdf')
                    print(f"Saved as PDF instead: {alt_path}")
                except Exception as e2:
                    print(f"All save attempts failed: {e2}")
                
        # Clean up
        fig.clear()
        gc.collect()
        
    except Exception as e:
        print(f"Error in combined visualization: {e}")
        import traceback
        traceback.print_exc()

def visualize_3d_data(X, y, title, save_path=None):
    """Visualize 3D data directly (no t-SNE)"""
    # Force garbage collection
    gc.collect()
    
    try:
        # Use direct Figure creation
        fig = matplotlib.figure.Figure(figsize=(10, 8))
        canvas = matplotlib.backends.backend_agg.FigureCanvasAgg(fig)
        
        # Add 3D axes
        ax = fig.add_subplot(111, projection='3d')
        
        # Convert to integer array
        y_plot = np.asarray(y, dtype=int)
        n_classes = len(np.unique(y_plot))
        
        # Plot each class separately
        for c in range(n_classes):
            mask = y_plot == c
            ax.scatter(
                X[mask, 0], X[mask, 1], X[mask, 2], 
                label=f"Class {c}", s=30, alpha=0.7
            )
        
        ax.set_title(title, fontsize=14)
        ax.legend()
        
        # Save figure
        if save_path:
            try:
                canvas.print_figure(save_path, dpi=100)
                print(f"3D figure saved to {save_path}")
            except Exception as e:
                print(f"Error saving 3D figure: {e}")
        
        # Clean up
        fig.clear()
        gc.collect()
        
    except Exception as e:
        print(f"Error in 3D visualization: {e}")
        import traceback
        traceback.print_exc()

class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features = None
        
        def hook(module, input, output):
            # 獲取第一個線性層的輸出特徵（256維）
            self.features = output.detach()
        
        # 獲取分類器中第一個線性層
        classifier_layers = list(self.model.classifier)
        print(classifier_layers)
        first_linear = classifier_layers[0]  # 第一個線性層
        
        if not isinstance(first_linear, nn.Linear):
            raise ValueError("無法找到分類器中的第一個線性層")
            
        # 註冊 hook 到第一個線性層
        first_linear.register_forward_hook(hook)
        
        print("Feature extractor attached to first linear layer (256-dim)")

def get_angle_from_class(class_name):
    """從類別名稱中提取角度值"""
    # 假設類別名稱格式為 'degXXX'，其中 XXX 是角度值
    try:
        return int(class_name[3:])  # 從 'deg' 後面提取數字
    except ValueError:
        return 0

def extract_features(model, dataset, device):
    """提取數據集中所有樣本的特徵"""
    model.eval()
    extractor = FeatureExtractor(model)
    features = []
    labels = []
    angles = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            data, label = dataset[i]
            data = data.unsqueeze(0).to(device)
            _ = model(data)  # 前向傳播
            
            # 檢查特徵維度
            if i == 0:  # 只在第一個樣本時打印
                print(f"Feature dimension: {extractor.features.shape}")
            
            features.append(extractor.features.cpu().numpy().reshape(-1))  # 展平特徵
            labels.append(label)
            
            # 從類別名稱獲取角度
            class_name = dataset.classes[label]
            angle = get_angle_from_class(class_name)
            angles.append(angle)
    
    return np.vstack(features), np.array(labels), np.array(angles)

def plot_tsne(features, angles, save_path):
    """使用t-SNE進行降維並繪製視覺化圖"""
    print("Starting t-SNE transformation...")
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(30, len(features) - 1),  # 確保 perplexity 小於樣本數
        n_iter=1000,
        verbose=1
    )
    features_tsne = tsne.fit_transform(features)
    
    print("Creating visualization...")
    plt.figure(figsize=(12, 8))
    
    # 使用散點圖繪製t-SNE結果
    scatter = plt.scatter(
        features_tsne[:, 0],
        features_tsne[:, 1],
        c=angles,
        cmap='viridis',
        s=100,
        alpha=0.6
    )
    
    plt.colorbar(scatter, label='Angle (degrees)')
    plt.title('t-SNE Visualization of CNN Last Layer Features', pad=20)
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 添加角度標籤
    for i, angle in enumerate(angles):
        plt.annotate(f'{angle}°', 
                    (features_tsne[i, 0], features_tsne[i, 1]),
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.7)
    
    # 保存圖形
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"t-SNE visualization saved to: {save_path}")

def main():
    print("Starting visualization process...")
    device = config.DEVICE
    print(f"Using device: {device}")
    
    # 讓使用者選擇頻率
    available_frequencies = ['500hz', '1000hz', '3000hz']
    print("\n可用頻率:")
    for i, freq in enumerate(available_frequencies):
        print(f"{i+1}. {freq}")
    
    while True:
        try:
            choice = int(input("\n請選擇頻率 (1-3): "))
            if 1 <= choice <= 3:
                selected_freq = available_frequencies[choice-1]
                break
            else:
                print("無效的選擇，請輸入1-3之間的數字")
        except ValueError:
            print("請輸入有效的數字")
    
    # 獲取指定頻率的最新模型文件
    model_files = [
        f for f in os.listdir(config.SAVE_DIR) 
        if f.startswith('simple_cnn_') 
        and selected_freq in f 
        and f.endswith('.pt')
    ]
    
    if not model_files:
        print(f"找不到頻率為 {selected_freq} 的模型文件！")
        return
    
    # 選擇最新的模型文件
    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(config.SAVE_DIR, x)))
    model_path = os.path.join(config.SAVE_DIR, latest_model)
    
    print(f"\nLoading model: {latest_model}")
    
    # 從文件名解析頻率和材質信息
    parts = latest_model.split('_')
    try:
        material = parts[2]
        frequency = parts[3]
    except IndexError:
        print("Error parsing model filename. Using default values.")
        material = "unknown"
        frequency = "unknown"
    
    # 加載數據集
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
    
    # 加載模型
    model = SimpleCNNAudioRanker(n_freqs=dataset.data.shape[2])
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print("Model architecture:")
    print(model)
    
    print("Extracting features...")
    features, labels, angles = extract_features(model, dataset, device)
    
    # 創建保存目錄
    plots_dir = os.path.join(config.SAVE_DIR, 'tsne_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 生成保存路徑
    save_path = os.path.join(plots_dir, f'tsne_visualization_{material}_{frequency}.png')
    
    # 繪製並保存t-SNE圖
    plot_tsne(features, angles, save_path)
    print("Visualization process completed!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
