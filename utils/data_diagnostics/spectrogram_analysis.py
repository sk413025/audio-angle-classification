"""
頻譜圖分析工具：用於檢查音頻頻譜圖的品質和特性
功能：
- 計算並報告頻譜圖的統計特性
- 繪製頻譜圖分佈的可視化圖表
- 檢測異常的頻譜圖樣本
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from sklearn.ensemble import IsolationForest
from scipy import stats
import os

def analyze_spectrogram_statistics(spectrograms, save_dir=None):
    """
    分析頻譜圖數據集的統計特性
    
    參數:
        spectrograms: 頻譜圖張量 [batch, channels, freq, time]
        save_dir: 保存結果的目錄路徑
    
    返回:
        stats_dict: 包含統計資訊的字典
    """
    if isinstance(spectrograms, torch.Tensor):
        spectrograms = spectrograms.detach().cpu().numpy()
    
    # 處理為 2D 數組以進行統計分析 [batch, features]
    batch_size = spectrograms.shape[0]
    orig_shape = spectrograms.shape
    flattened = spectrograms.reshape(batch_size, -1)
    
    # 計算基本統計量
    stats_dict = {
        'mean': np.mean(flattened, axis=1),
        'std': np.std(flattened, axis=1),
        'min': np.min(flattened, axis=1),
        'max': np.max(flattened, axis=1),
        'median': np.median(flattened, axis=1),
        'skew': stats.skew(flattened, axis=1),
        'kurtosis': stats.kurtosis(flattened, axis=1),
        'energy': np.sum(flattened**2, axis=1),
        'original_shape': orig_shape
    }
    
    # 計算全局統計量
    global_stats = {
        'global_mean': np.mean(flattened),
        'global_std': np.std(flattened),
        'global_min': np.min(flattened),
        'global_max': np.max(flattened),
        'global_median': np.median(flattened),
        'batch_size': batch_size,
        'non_zero_percentage': np.mean(flattened != 0) * 100,
        'freq_bins': orig_shape[2] if len(orig_shape) >= 3 else None,
        'time_bins': orig_shape[3] if len(orig_shape) >= 4 else None
    }
    stats_dict.update(global_stats)
    
    # 輸出摘要統計信息
    print(f"\n頻譜圖統計分析摘要 (批次大小: {batch_size}):")
    print(f"形狀: {orig_shape}")
    print(f"全局均值: {global_stats['global_mean']:.4f}")
    print(f"全局標準差: {global_stats['global_std']:.4f}")
    print(f"範圍: [{global_stats['global_min']:.4f}, {global_stats['global_max']:.4f}]")
    print(f"非零元素百分比: {global_stats['non_zero_percentage']:.2f}%")
    
    # 繪製與保存統計分佈圖
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 10))
        
        # 均值分佈
        plt.subplot(2, 2, 1)
        sns.histplot(stats_dict['mean'], kde=True)
        plt.title('頻譜圖均值分佈')
        plt.xlabel('均值')
        plt.ylabel('頻率')
        
        # 標準差分佈
        plt.subplot(2, 2, 2)
        sns.histplot(stats_dict['std'], kde=True)
        plt.title('頻譜圖標準差分佈')
        plt.xlabel('標準差')
        plt.ylabel('頻率')
        
        # 偏度分佈
        plt.subplot(2, 2, 3)
        sns.histplot(stats_dict['skew'], kde=True)
        plt.title('頻譜圖偏度分佈')
        plt.xlabel('偏度')
        plt.ylabel('頻率')
        
        # 能量分佈
        plt.subplot(2, 2, 4)
        sns.histplot(stats_dict['energy'], kde=True)
        plt.title('頻譜圖能量分佈')
        plt.xlabel('能量')
        plt.ylabel('頻率')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'spectrogram_statistics.png'))
        plt.close()
    
    return stats_dict

def plot_spectrogram_distribution(spectrograms, labels=None, class_names=None, save_dir=None):
    """
    繪製頻譜圖數據的分佈情況，可選按類別
    
    參數:
        spectrograms: 頻譜圖張量 [batch, channels, freq, time]
        labels: 類別標籤數組 (可選)
        class_names: 類別名稱列表 (可選)
        save_dir: 保存結果的目錄路徑
    """
    if isinstance(spectrograms, torch.Tensor):
        spectrograms = spectrograms.detach().cpu().numpy()
    
    batch_size = spectrograms.shape[0]
    
    # 計算每個頻譜圖的統計特徵
    features = []
    for i in range(batch_size):
        spec = spectrograms[i].reshape(-1)  # 將頻譜圖展平
        features.append([
            np.mean(spec),    # 均值
            np.std(spec),     # 標準差
            np.max(spec),     # 最大值
            np.sum(spec**2)   # 能量
        ])
    
    features = np.array(features)
    
    # 使用 TSNE 對高維特徵進行可視化
    if batch_size > 3:
        try:
            from sklearn.manifold import TSNE
            from sklearn.preprocessing import StandardScaler
            
            # 標準化特徵
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # 執行 TSNE 降維
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, batch_size-1))
            tsne_results = tsne.fit_transform(scaled_features)
            
            plt.figure(figsize=(10, 8))
            
            if labels is not None:
                unique_labels = np.unique(labels)
                for label in unique_labels:
                    idx = np.where(labels == label)[0]
                    label_name = class_names[label] if class_names is not None else f"Class {label}"
                    plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], label=label_name, alpha=0.7)
                plt.legend()
            else:
                plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.7)
            
            plt.title('頻譜圖特徵 t-SNE 分佈')
            plt.xlabel('t-SNE 維度 1')
            plt.ylabel('t-SNE 維度 2')
            
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, 'spectrogram_tsne.png'))
            
            plt.close()
            
        except Exception as e:
            print(f"TSNE 可視化失敗: {str(e)}")
    
    # 繪製頻譜圖特徵的分佈
    plt.figure(figsize=(16, 12))
    
    # 均值-標準差散點圖
    plt.subplot(2, 2, 1)
    if labels is not None:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            idx = np.where(labels == label)[0]
            label_name = class_names[label] if class_names is not None else f"Class {label}"
            plt.scatter(features[idx, 0], features[idx, 1], label=label_name, alpha=0.7)
        plt.legend()
    else:
        plt.scatter(features[:, 0], features[:, 1], alpha=0.7)
    plt.title('頻譜圖均值-標準差分佈')
    plt.xlabel('均值')
    plt.ylabel('標準差')
    
    # 均值-能量散點圖
    plt.subplot(2, 2, 2)
    if labels is not None:
        for label in unique_labels:
            idx = np.where(labels == label)[0]
            label_name = class_names[label] if class_names is not None else f"Class {label}"
            plt.scatter(features[idx, 0], features[idx, 3], label=label_name, alpha=0.7)
        plt.legend()
    else:
        plt.scatter(features[:, 0], features[:, 3], alpha=0.7)
    plt.title('頻譜圖均值-能量分佈')
    plt.xlabel('均值')
    plt.ylabel('能量')
    
    # 標準差-最大值散點圖
    plt.subplot(2, 2, 3)
    if labels is not None:
        for label in unique_labels:
            idx = np.where(labels == label)[0]
            label_name = class_names[label] if class_names is not None else f"Class {label}"
            plt.scatter(features[idx, 1], features[idx, 2], label=label_name, alpha=0.7)
        plt.legend()
    else:
        plt.scatter(features[:, 1], features[:, 2], alpha=0.7)
    plt.title('頻譜圖標準差-最大值分佈')
    plt.xlabel('標準差')
    plt.ylabel('最大值')
    
    # 頻譜圖平均值直方圖 (按類別)
    plt.subplot(2, 2, 4)
    if labels is not None:
        for label in unique_labels:
            idx = np.where(labels == label)[0]
            label_name = class_names[label] if class_names is not None else f"Class {label}"
            sns.kdeplot(features[idx, 0], label=label_name, fill=True, alpha=0.3)
        plt.legend()
    else:
        sns.histplot(features[:, 0], kde=True)
    plt.title('頻譜圖均值分佈')
    plt.xlabel('均值')
    plt.ylabel('密度')
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'spectrogram_feature_distribution.png'))
    
    plt.close()
    
    # 繪製頻譜圖平均模式
    # 計算每個類別的平均頻譜圖
    if labels is not None and spectrograms.shape[0] > 1:
        plt.figure(figsize=(15, 10))
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        
        # 計算每個頻道的網格佈局
        n_channels = spectrograms.shape[1]
        grid_size = int(np.ceil(np.sqrt(n_channels * n_classes)))
        
        for ch in range(n_channels):
            for i, label in enumerate(unique_labels):
                idx = np.where(labels == label)[0]
                class_specs = spectrograms[idx, ch]
                mean_spec = np.mean(class_specs, axis=0)
                
                # 計算子圖位置
                plt_idx = ch * n_classes + i + 1
                plt.subplot(grid_size, grid_size, plt_idx)
                
                img = plt.imshow(mean_spec, aspect='auto', origin='lower', norm=LogNorm())
                plt.colorbar(img, fraction=0.046, pad=0.04)
                
                label_name = class_names[label] if class_names is not None else f"Class {label}"
                plt.title(f'通道 {ch} - {label_name}')
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'spectrogram_class_means.png'))
        
        plt.close()
    
    return features

def detect_spectrogram_anomalies(spectrograms, contamination=0.05, save_dir=None):
    """
    檢測頻譜圖數據集中的異常樣本
    
    參數:
        spectrograms: 頻譜圖張量 [batch, channels, freq, time]
        contamination: 預期的異常比例 (0-0.5 之間)
        save_dir: 保存結果的目錄路徑
    
    返回:
        anomaly_indices: 異常樣本的索引數組
        anomaly_scores: 每個樣本的異常分數
    """
    if isinstance(spectrograms, torch.Tensor):
        spectrograms = spectrograms.detach().cpu().numpy()
    
    batch_size = spectrograms.shape[0]
    flattened = spectrograms.reshape(batch_size, -1)
    
    # 使用孤立森林檢測異常值
    clf = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    
    # 預測異常 (-1 為異常, 1 為正常)
    predictions = clf.fit_predict(flattened)
    anomaly_indices = np.where(predictions == -1)[0]
    
    # 計算異常分數 (負值表示異常可能性更大)
    anomaly_scores = clf.score_samples(flattened)
    normalized_scores = (anomaly_scores - np.min(anomaly_scores)) / (np.max(anomaly_scores) - np.min(anomaly_scores))
    normalized_scores = 1 - normalized_scores  # 轉換為 0-1，1 表示最異常
    
    # 輸出結果摘要
    n_anomalies = len(anomaly_indices)
    print(f"\n異常檢測結果:")
    print(f"總樣本數: {batch_size}")
    print(f"檢測到的異常數: {n_anomalies} ({n_anomalies/batch_size*100:.2f}%)")
    
    if n_anomalies > 0:
        print(f"異常樣本索引: {anomaly_indices}")
        print(f"前5個最異常樣本索引: {np.argsort(normalized_scores)[-5:][::-1]}")
    
    # 繪製異常分數分佈
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        sns.histplot(normalized_scores, bins=30, kde=True)
        if n_anomalies > 0:
            for idx in anomaly_indices:
                plt.axvline(x=normalized_scores[idx], color='r', linestyle='--', alpha=0.5)
        
        plt.title('頻譜圖異常分數分佈')
        plt.xlabel('異常分數 (越高表示越異常)')
        plt.ylabel('頻率')
        
        plt.savefig(os.path.join(save_dir, 'spectrogram_anomaly_scores.png'))
        plt.close()
        
        # 如果有異常樣本，繪製一些異常樣本
        if n_anomalies > 0:
            # 取最異常的幾個樣本
            n_to_plot = min(5, n_anomalies)
            top_anomalies = np.argsort(normalized_scores)[-n_to_plot:][::-1]
            
            plt.figure(figsize=(15, 3 * n_to_plot))
            
            for i, idx in enumerate(top_anomalies):
                plt.subplot(n_to_plot, 1, i+1)
                
                # 對多通道頻譜圖，只繪製第一個通道
                img = plt.imshow(spectrograms[idx, 0], aspect='auto', origin='lower', norm=LogNorm())
                plt.colorbar(img, fraction=0.046, pad=0.04)
                plt.title(f'異常樣本 #{idx}, 分數: {normalized_scores[idx]:.4f}')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'spectrogram_anomalies.png'))
            plt.close()
    
    return anomaly_indices, normalized_scores 