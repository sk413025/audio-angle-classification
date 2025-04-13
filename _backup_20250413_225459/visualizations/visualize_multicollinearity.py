"""
多重共線性分析工具
功能：
- 從模型中擷取特徵
- 計算特徵間相關矩陣
- 計算特徵矩陣的條件數
- 進行主成分分析（PCA）
- 視覺化特徵相關性與梯度行為
- 分析特徵值分布
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # 設置後端以避免顯示問題
import glob
import re
import scipy.linalg as la

# 添加父目錄到Python路徑
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 導入所需模組
import config
from datasets import SpectrogramDatasetWithMaterial, RankingPairDataset
from simple_cnn_models import ResNetAudioRanker

class FeatureExtractor:
    """
    特徵擷取器 - 從模型中擷取特定層的特徵
    """
    def __init__(self, model):
        """
        初始化特徵擷取器
        
        參數:
            model: 預訓練模型實例
        """
        self.model = model
        self.features = {}
        self.hooks = []
        
    def register_hook(self, name, module):
        """
        註冊特徵擷取的鉤子函數
        
        參數:
            name: 特徵名稱
            module: 要擷取特徵的模型層
        """
        def hook_fn(module, input, output):
            # 存儲該層的輸出特徵
            self.features[name] = output.detach().cpu()
            
        handle = module.register_forward_hook(hook_fn)
        self.hooks.append(handle)
        return handle
    
    def register_backbone_hooks(self):
        """
        為ResNet骨幹網絡註冊特徵擷取鉤子
        """
        self.register_hook('adapter_output', self.model.adapter)
        self.register_hook('visual_prompt_output', self.model.visual_prompt)
        self.register_hook('backbone_output', self.model.backbone)
        
        # 為分類器層添加hook
        if isinstance(self.model.classifier, torch.nn.Sequential):
            for i, layer in enumerate(self.model.classifier):
                if isinstance(layer, torch.nn.Linear):
                    self.register_hook(f'classifier_linear_{i}', layer)
    
    def extract_features(self, dataloader, device, max_batches=10):
        """
        從數據中擷取特徵
        
        參數:
            dataloader: 數據加載器
            device: 計算設備
            max_batches: 最大批次數量
            
        返回:
            特徵字典，包含各層特徵
        """
        self.model.eval()
        collected_features = {}
        
        batch_count = 0
        with torch.no_grad():
            for data1, data2, targets, _, _ in dataloader:
                if batch_count >= max_batches:
                    break
                    
                data1 = data1.to(device)
                self.features = {}  # 重置特徵字典
                
                # 前向傳播以觸發hook
                _ = self.model(data1)
                
                # 收集特徵
                for name, feature in self.features.items():
                    if name not in collected_features:
                        collected_features[name] = []
                    
                    # 如果是批次數據，展平為樣本列表
                    if len(feature.shape) >= 3:  # 如果是卷積層輸出
                        batch_size = feature.shape[0]
                        # 對每個樣本進行處理
                        for i in range(batch_size):
                            sample_feature = feature[i].view(-1).numpy()
                            collected_features[name].append(sample_feature)
                    else:  # 如果是全連接層輸出
                        collected_features[name].append(feature.view(-1).numpy())
                
                batch_count += 1
                
        # 將列表轉換為numpy數組
        for name in collected_features:
            collected_features[name] = np.array(collected_features[name])
            
        return collected_features
    
    def remove_hooks(self):
        """移除所有鉤子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

class MulticollinearityAnalyzer:
    """
    多重共線性分析器 - 分析特徵間的相關性和條件數
    增加先降維再分析的功能
    """
    def __init__(self):
        """初始化多重共線性分析器"""
        self.original_features = None
        self.original_feature_names = None
        self.reduced_features = None  # 儲存降維後的特徵
        self.pca_model = None         # 儲存PCA模型
        self.analysis_features = None # 當前用於分析的特徵 (可能是原始或降維後的)
        self.feature_names = None     # 當前用於分析的特徵名稱
        self.correlation_matrix = None
        self.condition_number = None
        self.pca_on_original_results = None # PCA on original features
        self.eigenvalues = None
        
    def load_features(self, features_dict, layer_name=None):
        """
        載入原始特徵進行分析
        
        參數:
            features_dict: 特徵字典，包含各層特徵
            layer_name: 指定要分析的層名稱，如果為None則使用backbone_output
        """
        if layer_name is None:
            layer_name = 'backbone_output'
            
        if layer_name not in features_dict:
            raise ValueError(f"特徵字典中找不到層 '{layer_name}'")
            
        self.original_features = features_dict[layer_name]
        self.original_feature_names = [f"Feature_{i}" for i in range(self.original_features.shape[1])]
        
        # 默認使用原始特徵進行分析
        self.analysis_features = self.original_features
        self.feature_names = self.original_feature_names
        
        print(f"Loaded features from layer '{layer_name}', shape: {self.original_features.shape}")
        return self.original_features
        
    def reduce_features_with_pca(self, variance_threshold=0.95):
        """
        使用PCA對原始特徵進行降維，並將降維後的特徵設為當前分析目標
        
        參數:
            variance_threshold: 保留的累積方差比例閾值
        """
        if self.original_features is None:
            raise ValueError("請先載入原始特徵")

        print(f"Reducing original features using PCA (keeping {variance_threshold*100:.0f}% variance)...")
        
        # 1. 標準化原始特徵
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.original_features)
        
        # 2. 執行PCA以確定保留的維度
        pca_full = PCA()
        pca_full.fit(features_scaled)
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        print(f"  - Original dimension: {features_scaled.shape[1]}")
        print(f"  - Reduced dimension: {n_components}")
        
        # 3. 執行PCA並轉換數據
        self.pca_model = PCA(n_components=n_components)
        self.reduced_features = self.pca_model.fit_transform(features_scaled)
        
        # 4. 更新當前分析目標為降維後的特徵
        self.analysis_features = self.reduced_features
        self.feature_names = [f"PC_{i+1}" for i in range(n_components)] # 主成分名稱
        self.pca_on_original_results = { # Store results from PCA on original data
            'explained_variance_ratio': pca_full.explained_variance_ratio_,
            'explained_variance': pca_full.explained_variance_,
            'cumulative_variance': cumulative_variance
        }
        self.eigenvalues = pca_full.explained_variance_ # Store eigenvalues from original PCA
        
        print(f"  - PCA dimension reduction complete. Current analysis features shape: {self.analysis_features.shape}")
        return self.reduced_features

    def use_original_features(self):
        """將分析目標切換回原始特徵"""
        if self.original_features is None:
             raise ValueError("請先載入原始特徵")
        self.analysis_features = self.original_features
        self.feature_names = self.original_feature_names
        print(f"Switched analysis target back to original features, shape: {self.analysis_features.shape}")
        
    def compute_correlation_matrix(self):
        """
        計算當前分析特徵間的相關矩陣
        
        返回:
            相關矩陣
        """
        if self.analysis_features is None:
            raise ValueError("請先載入或降維特徵")
            
        print(f"Calculating correlation matrix (based on {len(self.feature_names)} features)...")
        # 注意：PCA輸出的主成分已經是標準化的，且理論上不相關，但仍可計算檢查
        self.correlation_matrix = np.corrcoef(self.analysis_features, rowvar=False)
        
        # 處理單一維度降維後結果 (返回 [[1.]])
        if self.correlation_matrix.ndim == 0: 
             self.correlation_matrix = np.array([[1.0]])
        elif self.correlation_matrix.ndim == 1: # Should not happen with corrcoef but for safety
             self.correlation_matrix = np.array([[1.0]])

        print(f"Correlation matrix calculation complete, shape: {self.correlation_matrix.shape}")
        return self.correlation_matrix
    
    def compute_condition_number(self):
        """
        計算當前分析特徵矩陣的條件數
        
        返回:
            條件數
        """
        if self.analysis_features is None:
            raise ValueError("請先載入或降維特徵")
            
        print(f"Calculating condition number (based on {len(self.feature_names)} features)...")
            
        # PCA輸出的主成分已經是標準化的
        features_for_cond = self.analysis_features
        if features_for_cond.shape[1] == 0:
            print("Warning: Feature dimension is 0, cannot calculate condition number")
            self.condition_number = np.nan
            return self.condition_number
        if features_for_cond.shape[1] == 1:
             print("Warning: Feature dimension is 1, condition number is defined as 1")
             self.condition_number = 1.0
             return self.condition_number

        # 計算奇異值
        try:
            s = la.svdvals(features_for_cond)
            # 確保有足夠的奇異值
            if len(s) < 1 or s[-1] < 1e-12: # Avoid division by near zero
                print("Warning: Minimum singular value is too small or zero. Condition number might be infinite.")
                self.condition_number = np.inf
            else:
                self.condition_number = s[0] / s[-1]
        except la.LinAlgError:
            print("Warning: SVD computation failed. Setting condition number to infinity.")
            self.condition_number = np.inf
        
        print(f"Feature matrix condition number: {self.condition_number:.2f}")
        if np.isinf(self.condition_number) or self.condition_number > 100:
            print("Warning: Condition number is extremely large or > 100, indicating severe multicollinearity.")
        elif self.condition_number > 30:
            print("Note: Condition number > 30, indicating moderate multicollinearity.")
            
        return self.condition_number
    
    def perform_pca_analysis_on_original(self):
        """
        對原始特徵進行主成分分析（主要用於獲取特徵值分布）
        
        返回:
            PCA分析原始特徵的結果字典
        """
        if self.original_features is None:
            raise ValueError("請先載入原始特徵")
        
        if self.pca_on_original_results is None:
             print("Performing PCA on original features to analyze eigenvalue distribution...")
             # 標準化原始特徵
             scaler = StandardScaler()
             features_scaled = scaler.fit_transform(self.original_features)
             
             # 執行完整PCA以獲取所有特徵值
             pca_full = PCA()
             pca_full.fit(features_scaled)
             self.eigenvalues = pca_full.explained_variance_
             cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
             
             self.pca_on_original_results = {
                 'explained_variance_ratio': pca_full.explained_variance_ratio_,
                 'explained_variance': pca_full.explained_variance_,
                 'cumulative_variance': cumulative_variance
             }
             print("  - PCA analysis on original features complete.")
             
        # 計算95%方差所需的主成分數量
        cumulative_variance = self.pca_on_original_results['cumulative_variance']
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        
        print(f"PCA Analysis (based on original {self.original_features.shape[1]} features)")
        print(f"  - Principal components needed for 95% variance: {n_components_95}")
        print(f"  - Max Eigenvalue: {self.eigenvalues[0]:.2f}")
        print(f"  - Min Eigenvalue: {self.eigenvalues[-1]:.2f}")
        # Avoid division by zero if min eigenvalue is effectively zero
        min_eig = self.eigenvalues[-1]
        ratio = self.eigenvalues[0] / min_eig if min_eig > 1e-12 else np.inf
        print(f"  - Eigenvalue Ratio (Max/Min): {ratio:.2f}")
        
        return self.pca_on_original_results
    
    def find_highly_correlated_features(self, threshold=0.9):
        """
        在當前分析特徵中尋找高度相關的特徵對
        
        參數:
            threshold: 相關係數閾值
            
        返回:
            高度相關的特徵對列表
        """
        if self.correlation_matrix is None:
            self.compute_correlation_matrix()
        
        # 如果相關矩陣只有一個元素 (降維到1維) 或為空
        if self.correlation_matrix.size <= 1:
             print(f"Feature dimension is too low ({self.correlation_matrix.shape[0]}), cannot find correlated pairs.")
             return []
             
        highly_correlated = []
        n_features = self.correlation_matrix.shape[0]
        for i in range(n_features):
            for j in range(i+1, n_features):
                # 檢查索引是否有效
                if i < self.correlation_matrix.shape[0] and j < self.correlation_matrix.shape[1]:
                    corr = self.correlation_matrix[i, j]
                    if abs(corr) >= threshold:
                        highly_correlated.append((i, j, corr))
                    
        highly_correlated.sort(key=lambda x: abs(x[2]), reverse=True)
        
        print(f"Found {len(highly_correlated)} highly correlated feature pairs (|correlation| >= {threshold}, based on {n_features} features)")
        
        return highly_correlated

class MulticollinearityVisualizer:
    """
    多重共線性視覺化工具 - 生成各種視覺化圖表
    """
    def __init__(self, analyzer):
        """
        初始化視覺化工具
        
        參數:
            analyzer: MulticollinearityAnalyzer實例
        """
        self.analyzer = analyzer
        
    def plot_correlation_heatmap(self, save_path=None, feature_indices=None, show_values=True):
        """
        繪製特徵相關性熱力圖
        
        參數:
            save_path: 圖片保存路徑
            feature_indices: 要顯示的特徵索引列表，如果為None則顯示所有特徵
            show_values: 是否在熱力圖上顯示數值
            
        返回:
            圖片保存路徑
        """
        if self.analyzer.correlation_matrix is None:
            self.analyzer.compute_correlation_matrix()
            
        corr_matrix = self.analyzer.correlation_matrix
        feature_names = self.analyzer.feature_names
        
        # 如果指定了特徵索引，則只顯示這些特徵
        if feature_indices is not None:
            corr_matrix = corr_matrix[np.ix_(feature_indices, feature_indices)]
            feature_names = [feature_names[i] for i in feature_indices]
            
        # 繪製熱力圖
        plt.figure(figsize=(12, 10))
        mask = np.zeros_like(corr_matrix)
        mask[np.triu_indices_from(mask)] = True  # 上三角形為True
        
        # 使用seaborn繪製美觀的熱力圖
        ax = sns.heatmap(
            corr_matrix, 
            annot=show_values,  # 顯示數值
            fmt=".2f" if show_values else "", 
            cmap="coolwarm",
            vmin=-1, vmax=1,
            mask=mask,
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .8},
            xticklabels=feature_names if len(feature_names) < 30 else False,
            yticklabels=feature_names if len(feature_names) < 30 else False
        )
        
        plt.title("Feature Correlation Heatmap", fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Correlation heatmap saved to: {save_path}")
            
        plt.close()
        return save_path
    
    def plot_eigenvalue_distribution(self, save_path=None):
        """
        繪製特徵值分布圖
        
        參數:
            save_path: 圖片保存路徑
            
        返回:
            圖片保存路徑
        """
        if self.analyzer.pca_on_original_results is None:
            self.analyzer.perform_pca_analysis_on_original()
            
        if self.analyzer.eigenvalues is None:
            raise ValueError("無法獲取用於繪圖的特徵值，請確保原始PCA已執行")
        
        eigenvalues = self.analyzer.eigenvalues
        pca_results = self.analyzer.pca_on_original_results
        
        # 繪製特徵值和累積解釋方差
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 特徵值分布圖
        ax1.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'o-', markersize=8)
        ax1.set_title('Eigenvalue Distribution (Scree Plot)', fontsize=14)
        ax1.set_xlabel('Principal Component', fontsize=12)
        ax1.set_ylabel('Eigenvalue', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 標記前5個主成分
        for i in range(min(5, len(eigenvalues))):
            ax1.annotate(f'{eigenvalues[i]:.2f}', 
                        xy=(i+1, eigenvalues[i]), 
                        xytext=(5, 5),
                        textcoords='offset points')
        
        # 累積解釋方差圖
        explained_variance_ratio = pca_results['explained_variance_ratio']
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'o-', markersize=8)
        ax2.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='95% Variance')
        
        # 找出達到95%方差的主成分數量
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        ax2.axvline(x=n_components_95, color='g', linestyle='--', alpha=0.5, 
                   label=f'{n_components_95} Principal Components')
        
        ax2.set_title('Cumulative Explained Variance Ratio', fontsize=14)
        ax2.set_xlabel('Number of Principal Components', fontsize=12)
        ax2.set_ylabel('Cumulative Explained Variance Ratio', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 標記達到特定比例的點
        for threshold in [0.5, 0.8, 0.9, 0.95, 0.99]:
            valid_indices = np.where(cumulative_variance >= threshold)[0]
            if len(valid_indices) > 0:
                n_comp = valid_indices[0] + 1
                ax2.annotate(f'{threshold:.0%}: {n_comp} Components', 
                            xy=(n_comp, threshold), 
                            xytext=(5, -10 if threshold > 0.9 else 5),
                            textcoords='offset points')
            else:
                pass 
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Eigenvalue distribution plot saved to: {save_path}")
            
        plt.close()
        return save_path
    
    def plot_high_correlation_network(self, threshold=0.9, save_path=None):
        """
        繪製高相關特徵的網絡圖
        
        參數:
            threshold: 相關係數閾值
            save_path: 圖片保存路徑
            
        返回:
            圖片保存路徑
        """
        # 獲取高度相關的特徵對
        correlated_pairs = self.analyzer.find_highly_correlated_features(threshold)
        
        if not correlated_pairs:
            print(f"未找到相關係數 >= {threshold} 的特徵對")
            return None
            
        # 嘗試導入networkx，若失敗則返回None
        try:
            import networkx as nx
            
            # 創建網絡圖
            G = nx.Graph()
            
            # 添加節點和邊
            for i, j, corr in correlated_pairs:
                feature1 = self.analyzer.feature_names[i]
                feature2 = self.analyzer.feature_names[j]
                
                G.add_node(feature1)
                G.add_node(feature2)
                G.add_edge(feature1, feature2, weight=abs(corr))
            
            # 繪製網絡圖
            plt.figure(figsize=(12, 12))
            
            # 使用權重確定位置
            pos = nx.spring_layout(G, weight='weight')
            
            # 繪製節點
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                  node_size=300, alpha=0.8)
            
            # 繪製邊，使用相關係數作為邊的粗細
            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]
            nx.draw_networkx_edges(G, pos, edgelist=edges, width=[w*3 for w in weights], 
                                  alpha=0.7, edge_color='darkblue')
            
            # 繪製標籤
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
            
            plt.title(f"Feature Correlation Network (Threshold: {threshold})", fontsize=16)
            plt.axis('off')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Feature correlation network saved to: {save_path}")
                
            plt.close()
            return save_path
            
        except ImportError:
            print("繪製網絡圖需要安裝networkx庫")
            return None
    
    def plot_condition_number_comparison(self, condition_numbers, labels, save_path=None):
        """
        繪製不同檢查點的條件數比較圖
        
        參數:
            condition_numbers: 條件數列表
            labels: 對應的標籤列表（例如epoch數）
            save_path: 圖片保存路徑
            
        返回:
            圖片保存路徑
        """
        plt.figure(figsize=(10, 6))
        
        bars = plt.bar(range(len(condition_numbers)), condition_numbers, alpha=0.7)
        
        # 添加條件數警戒線
        plt.axhline(y=30, color='orange', linestyle='--', alpha=0.7, label='Medium Multicollinearity (30)')
        plt.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Severe Multicollinearity (100)')
        
        # 設置x軸標籤
        plt.xticks(range(len(labels)), labels, rotation=45)
        
        # 在條形上方顯示條件數值
        for i, v in enumerate(condition_numbers):
            plt.text(i, v + 5, f'{v:.1f}', ha='center')
            
        # 根據條件數著色
        for i, bar in enumerate(bars):
            if condition_numbers[i] > 100:
                bar.set_color('red')
            elif condition_numbers[i] > 30:
                bar.set_color('orange')
            else:
                bar.set_color('green')
        
        plt.title('Condition Number Comparison', fontsize=16)
        plt.xlabel('Training Stage', fontsize=12)
        plt.ylabel('Condition Number', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Condition number comparison plot saved to: {save_path}")
            
        plt.close()
        return save_path

class GradientCorrelationAnalyzer:
    """
    梯度相關性分析器 - 分析梯度與特徵相關性的關係
    """
    def __init__(self, model, criterion, dataloader, device):
        """
        初始化梯度相關性分析器
        
        參數:
            model: 預訓練模型
            criterion: 損失函數
            dataloader: 數據加載器
            device: 計算設備
        """
        self.model = model
        self.criterion = criterion
        self.dataloader = dataloader
        self.device = device
        self.gradient_data = None
        
    def collect_gradients(self, num_batches=10):
        """
        收集多個批次的梯度數據
        
        參數:
            num_batches: 要收集的批次數量
            
        返回:
            收集到的梯度數據
        """
        self.model.train()  # 設置為訓練模式以啟用梯度計算
        gradient_samples = []
        
        for i, (data1, data2, targets, _, _) in enumerate(self.dataloader):
            if i >= num_batches:
                break
                
            data1, data2, targets = data1.to(self.device), data2.to(self.device), targets.to(self.device)
            
            # 重置梯度
            self.model.zero_grad()
            
            # 前向傳播
            outputs1 = self.model(data1)
            outputs2 = self.model(data2)
            
            # 確保維度一致
            outputs1 = outputs1.view(-1)
            outputs2 = outputs2.view(-1)
            targets = targets.view(-1)
            
            # 計算損失
            loss = self.criterion(outputs1, outputs2, targets)
            
            # 反向傳播
            loss.backward()
            
            # 收集梯度
            batch_gradients = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    batch_gradients[name] = param.grad.detach().cpu().numpy().flatten()
            
            gradient_samples.append(batch_gradients)
            
        self.gradient_data = gradient_samples
        print(f"已收集 {len(gradient_samples)} 個批次的梯度數據")
        return gradient_samples
    
    def analyze_gradient_oscillation(self, mc_analyzer, threshold=0.9):
        """
        分析高相關特徵的梯度振盪
        
        參數:
            mc_analyzer: MulticollinearityAnalyzer實例
            threshold: 相關係數閾值
            
        返回:
            梯度振盪分析結果
        """
        if self.gradient_data is None:
            raise ValueError("請先收集梯度數據")
            
        if mc_analyzer.correlation_matrix is None:
            raise ValueError("請先計算特徵相關矩陣")
            
        # 獲取高相關特徵對
        correlated_pairs = mc_analyzer.find_highly_correlated_features(threshold)
        
        if not correlated_pairs:
            print(f"未找到相關係數 >= {threshold} 的特徵對")
            return None
            
        # 找出帶有權重的層
        weight_layers = []
        for name, param in self.model.named_parameters():
            if 'weight' in name and len(param.shape) == 2:  # 只考慮2D權重矩陣
                weight_layers.append(name)
        
        oscillation_results = {}
        
        for layer_name in weight_layers:
            # 從所有批次中提取該層的梯度
            layer_gradients = np.array([sample[layer_name] for sample in self.gradient_data 
                                       if layer_name in sample])
            
            if len(layer_gradients) == 0:
                continue
                
            # 計算梯度方向的余弦相似度
            cosine_similarities = np.zeros((len(layer_gradients), len(layer_gradients)))
            for i in range(len(layer_gradients)):
                for j in range(len(layer_gradients)):
                    dot_product = np.dot(layer_gradients[i], layer_gradients[j])
                    norm_i = np.linalg.norm(layer_gradients[i])
                    norm_j = np.linalg.norm(layer_gradients[j])
                    
                    if norm_i > 0 and norm_j > 0:
                        cosine_similarities[i, j] = dot_product / (norm_i * norm_j)
                    else:
                        cosine_similarities[i, j] = 0
            
            # 計算梯度方向的平均余弦相似度
            mean_cosine = np.mean(cosine_similarities)
            
            # 計算梯度方向的標準差
            std_cosine = np.std(cosine_similarities)
            
            oscillation_results[layer_name] = {
                'mean_cosine': mean_cosine,
                'std_cosine': std_cosine,
                'cosine_similarities': cosine_similarities
            }
            
        return oscillation_results
    
    def plot_gradient_oscillation(self, oscillation_results, save_path=None):
        """
        繪製梯度振盪分析圖
        
        參數:
            oscillation_results: 梯度振盪分析結果
            save_path: 圖片保存路徑
            
        返回:
            圖片保存路徑
        """
        if not oscillation_results:
            print("沒有梯度振盪分析結果可繪製")
            return None
            
        n_layers = len(oscillation_results)
        plt.figure(figsize=(12, 4 * n_layers))
        
        for i, (layer_name, results) in enumerate(oscillation_results.items()):
            # 在多個子圖中顯示每層的結果
            ax = plt.subplot(n_layers, 1, i+1)
            
            # 繪製余弦相似度矩陣
            im = ax.imshow(results['cosine_similarities'], cmap='coolwarm', vmin=-1, vmax=1)
            
            ax.set_title(f"{layer_name} - Gradient Direction Cosine Similarity\nMean: {results['mean_cosine']:.3f}, Std: {results['std_cosine']:.3f}")
            plt.colorbar(im, ax=ax)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gradient oscillation analysis plot saved to: {save_path}")
            
        plt.close()
        return save_path 

def find_checkpoint_files(material, frequency):
    """
    查找指定材質和頻率的所有檢查點文件
    
    參數:
        material: 材質名稱
        frequency: 頻率名稱
    
    返回:
        檢查點文件列表
    """
    checkpoint_dir = os.path.join(config.SAVE_DIR, 'model_checkpoints', f"{material}_{frequency}")
    
    if not os.path.exists(checkpoint_dir):
        return []
    
    # 查找所有檢查點文件
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_epoch_*.pt"))
    
    # 提取epoch信息並排序
    def get_epoch(filename):
        match = re.search(r'model_epoch_(\d+)_', os.path.basename(filename))
        if match:
            return int(match.group(1))
        return 0
    
    checkpoint_files.sort(key=get_epoch)
    return checkpoint_files

def analyze_model_checkpoint(checkpoint_path, dataset, device, pca_variance_threshold=0.95):
    """
    分析單個模型檢查點的多重共線性 (先使用PCA降維)
    
    參數:
        checkpoint_path: 檢查點文件路徑
        dataset: 數據集
        device: 計算設備
        pca_variance_threshold: PCA降維保留的方差比例
    
    返回:
        分析結果
    """
    print(f"\nAnalyzing checkpoint: {os.path.basename(checkpoint_path)}")
    
    # 加載檢查點
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 創建排序對數據集
    ranking_dataset = RankingPairDataset(dataset)
    print(f"Number of ranking pairs: {len(ranking_dataset)}")
    
    # 創建數據加載器
    # 降低 num_workers 避免潛在的 MPS 問題
    num_workers = 0 if str(device) == 'mps' else 2 
    batch_size = min(32, len(ranking_dataset))
    dataloader = DataLoader(
        ranking_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers, 
        drop_last=True
    )
    
    # 初始化模型並加載權重
    model = ResNetAudioRanker(n_freqs=dataset.data.shape[2])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # 損失函數
    criterion = torch.nn.MarginRankingLoss(margin=config.MARGIN)
    
    # 提取特徵
    print("Extracting model features...")
    extractor = FeatureExtractor(model)
    extractor.register_backbone_hooks()
    features_dict = extractor.extract_features(dataloader, device)
    extractor.remove_hooks()
    
    # 多重共線性分析
    print("Performing multicollinearity analysis...")
    mc_analyzer = MulticollinearityAnalyzer()
    
    # 載入原始特徵
    mc_analyzer.load_features(features_dict, 'backbone_output')
    
    # *** 新增：執行PCA降維 ***
    mc_analyzer.reduce_features_with_pca(variance_threshold=pca_variance_threshold)
    
    # *** 修改：在降維後的特徵上計算相關性和條件數 ***
    mc_analyzer.compute_correlation_matrix()  # Now operates on reduced features
    condition_number = mc_analyzer.compute_condition_number() # Now operates on reduced features
    
    # *** 修改：PCA分析現在只用於視覺化原始特徵的特徵值分布 ***
    pca_original_results = mc_analyzer.perform_pca_analysis_on_original()
    
    # *** 修改：在降維後的特徵上查找高相關對 ***
    highly_correlated = mc_analyzer.find_highly_correlated_features(threshold=0.9)
    
    # 獲取epoch號碼用於文件名
    epoch = checkpoint['epoch']
    
    # 創建保存目錄
    material = config.MATERIAL
    frequency = dataset.selected_freq
    plots_dir = os.path.join(config.SAVE_DIR, 'multicollinearity_plots', f"{material}_{frequency}_pca_{pca_variance_threshold*100:.0f}") # 加入PCA閾值到目錄名
    os.makedirs(plots_dir, exist_ok=True)
    
    # 創建視覺化 (分析器實例現在包含原始和降維結果)
    print("Generating visualization plots...")
    visualizer = MulticollinearityVisualizer(mc_analyzer)
    
    # 1. 繪製 *降維後* 特徵的相關性熱力圖
    heatmap_path = os.path.join(plots_dir, f'correlation_heatmap_reduced_epoch_{epoch}.png')
    visualizer.plot_correlation_heatmap(heatmap_path, show_values=(mc_analyzer.analysis_features.shape[1] < 30)) # 降維後維度少於30才顯示數值
    
    # 2. 繪製 *原始* 特徵的特徵值分布圖 (來自 perform_pca_analysis_on_original)
    eigenvalue_path = os.path.join(plots_dir, f'eigenvalue_distribution_original_epoch_{epoch}.png')
    visualizer.plot_eigenvalue_distribution(eigenvalue_path) # This uses eigenvalues from original PCA
    
    # 3. 繪製 *降維後* 特徵的高相關網絡圖
    network_path = os.path.join(plots_dir, f'correlation_network_reduced_epoch_{epoch}.png')
    visualizer.plot_high_correlation_network(threshold=0.9, save_path=network_path)
    
    # 梯度分析 (仍然基於原始模型計算梯度，但可能與降維後的特徵相關性分析結果結合解讀)
    print("Performing gradient correlation analysis...")
    grad_analyzer = GradientCorrelationAnalyzer(model, criterion, dataloader, device)
    grad_analyzer.collect_gradients(num_batches=5)
    # 注意：梯度振盪分析目前沒有直接使用降維後的特徵，它分析的是權重梯度
    # 但我們可以與降維後發現的低相關性進行對比
    oscillation_results = grad_analyzer.analyze_gradient_oscillation(mc_analyzer)
    
    # 梯度振盪視覺化
    if oscillation_results:
        grad_path = os.path.join(plots_dir, f'gradient_oscillation_epoch_{epoch}.png')
        grad_analyzer.plot_gradient_oscillation(oscillation_results, grad_path)
    
    print(f"Analysis for checkpoint {epoch} complete!")
    
    return {
        'epoch': epoch,
        'condition_number_reduced': condition_number, # 條件數是基於降維後特徵計算的
        'highly_correlated_count_reduced': len(highly_correlated), # 高相關對是基於降維後特徵計算的
        'original_dim': mc_analyzer.original_features.shape[1],
        'reduced_dim': mc_analyzer.analysis_features.shape[1],
        'pca_n_components_95_original': np.argmax(np.cumsum(pca_original_results['explained_variance_ratio']) >= 0.95) + 1
    }

def main():
    """主函數"""
    print("Starting multicollinearity analysis (with PCA preprocessing)...")
    device = config.DEVICE
    print(f"Using device: {device}")
    
    # 讓用戶選擇頻率
    available_frequencies = ['500hz', '1000hz', '3000hz']
    print("\nAvailable frequencies:")
    for i, freq in enumerate(available_frequencies):
        print(f"{i+1}. {freq}")
    
    while True:
        try:
            choice = int(input(f"\nSelect frequency (1-{len(available_frequencies)}): "))
            if 1 <= choice <= len(available_frequencies):
                selected_freq = available_frequencies[choice-1]
                break
            else:
                print(f"Invalid choice, please enter a number between 1 and {len(available_frequencies)}")
        except ValueError:
            print("Please enter a valid number")
            
    # 新增：讓用戶選擇PCA方差閾值
    while True:
        try:
             pca_threshold_str = input("\nEnter PCA variance threshold (e.g., 0.95, 0.99): [Default 0.95] ") or "0.95"
             pca_variance_threshold = float(pca_threshold_str)
             if 0.5 <= pca_variance_threshold <= 1.0:
                 break
             else:
                 print("Threshold must be between 0.5 and 1.0")
        except ValueError:
            print("Please enter a valid decimal number")

    # 加載數據集
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
        
    # 查找所有檢查點
    checkpoint_files = find_checkpoint_files(config.MATERIAL, selected_freq)
    
    if not checkpoint_files:
        print(f"No checkpoint files found for {config.MATERIAL}_{selected_freq}")
        return
    
    print(f"\nAvailable checkpoint files: {len(checkpoint_files)}")
    for i, path in enumerate(checkpoint_files):
        match = re.search(r'model_epoch_(\d+)_', os.path.basename(path))
        epoch = match.group(1) if match else "Unknown"
        print(f"{i+1}. Epoch {epoch}")
        
    # 選擇檢查點
    while True:
        try:
            choice = input(f"\nSelect checkpoint to analyze (1-{len(checkpoint_files)}, or 'all' for all): ")
            if choice.lower() == 'all':
                selected_checkpoints = checkpoint_files
                break
            else:
                idx = int(choice) - 1
                if 0 <= idx < len(checkpoint_files):
                    selected_checkpoints = [checkpoint_files[idx]]
                    break
                else:
                    print(f"Invalid choice, please enter a number between 1 and {len(checkpoint_files)}")
        except ValueError:
            print("Please enter a valid number or 'all'")
            
    # 分析所選檢查點
    analysis_results = []
    for checkpoint_path in selected_checkpoints:
        # *** 傳遞PCA閾值 ***
        result = analyze_model_checkpoint(checkpoint_path, dataset, device, pca_variance_threshold)
        analysis_results.append(result)
    
    # 如果分析了多個檢查點，繪製比較圖
    if len(analysis_results) > 1:
        print("\nGenerating comparison plots...")
        # 修改輸出目錄名以包含PCA閾值
        plots_dir = os.path.join(config.SAVE_DIR, 'multicollinearity_plots', f"{config.MATERIAL}_{selected_freq}_pca_{pca_variance_threshold*100:.0f}")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 條件數比較 (基於降維後特徵)
        condition_numbers = [result.get('condition_number_reduced', np.nan) for result in analysis_results]
        epochs = [f"Epoch {result.get('epoch', '?')}" for result in analysis_results]
        
        visualizer = MulticollinearityVisualizer(None)  # 只用於繪圖
        condition_path = os.path.join(plots_dir, 'condition_number_comparison_reduced.png')
        visualizer.plot_condition_number_comparison(condition_numbers, epochs, condition_path)
    
    print("\nMulticollinearity analysis complete!")

if __name__ == "__main__":
    main() 