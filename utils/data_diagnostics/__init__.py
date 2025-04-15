"""
數據診斷工具：用於分析和可視化频谱图數據集
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

def analyze_spectrogram_statistics(data, save_dir=None):
    """分析頻譜圖統計特性"""
    print("分析頻譜圖統計特性...")
    
    # 計算基本統計量
    mean_val = torch.mean(data).item()
    std_val = torch.std(data).item()
    min_val = torch.min(data).item()
    max_val = torch.max(data).item()
    
    print(f"平均值: {mean_val:.4f}")
    print(f"標準差: {std_val:.4f}")
    print(f"最小值: {min_val:.4f}")
    print(f"最大值: {max_val:.4f}")
    
    # 保存結果
    if save_dir:
        plt.figure(figsize=(10, 6))
        plt.hist(data.view(-1).cpu().numpy(), bins=50)
        plt.title('Spectrogram Pixel Distribution')
        plt.xlabel('Pixel Values')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(save_dir, 'spectrogram_distribution.png'))
        plt.close()
        
        # 保存統計信息到文本文件
        with open(os.path.join(save_dir, 'spectrogram_stats.txt'), 'w') as f:
            f.write(f"Mean: {mean_val:.4f}\n")
            f.write(f"Std Dev: {std_val:.4f}\n")
            f.write(f"Min Value: {min_val:.4f}\n")
            f.write(f"Max Value: {max_val:.4f}\n")
    
    return {
        'mean': mean_val,
        'std': std_val,
        'min': min_val,
        'max': max_val
    }

def plot_spectrogram_distribution(data, labels, class_names, save_dir=None):
    """繪製頻譜圖分佈"""
    print("繪製頻譜圖分佈...")
    
    # 確保數據在CPU上並轉換為NumPy
    data_np = data.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # 對每個類別計算平均頻譜圖
    unique_labels = np.unique(labels_np)
    
    if save_dir:
        plt.figure(figsize=(15, 10))
        for i, label in enumerate(unique_labels):
            mask = (labels_np == label)
            if np.sum(mask) > 0:
                mean_spectrogram = np.mean(data_np[mask], axis=0)
                
                # 顯示平均頻譜圖
                plt.subplot(len(unique_labels), 1, i+1)
                plt.imshow(mean_spectrogram[0], aspect='auto', origin='lower')
                plt.colorbar(format='%+2.0f dB')
                plt.title(f'Average Spectrogram for Class {class_names[label]}')
                plt.ylabel('Frequency')
                
                if i == len(unique_labels) - 1:
                    plt.xlabel('Time')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'class_spectrograms.png'))
        plt.close()
    
    return True

def detect_spectrogram_anomalies(data, contamination=0.05, save_dir=None):
    """檢測異常頻譜圖"""
    print("檢測異常頻譜圖...")
    
    # 準備數據 (展平頻譜圖)
    data_flat = data.view(data.shape[0], -1).cpu().numpy()
    
    # 使用隔離森林檢測異常
    clf = IsolationForest(contamination=contamination, random_state=42)
    outlier_pred = clf.fit_predict(data_flat)
    
    # -1 表示異常，1 表示正常
    outliers = np.where(outlier_pred == -1)[0]
    print(f"檢測到 {len(outliers)} 個異常樣本（總共 {len(data_flat)} 個樣本）")
    
    # 保存結果
    if save_dir and len(outliers) > 0:
        plt.figure(figsize=(15, 5 * min(len(outliers), 5)))
        for i, idx in enumerate(outliers[:5]):  # 只顯示前5個異常
            plt.subplot(min(len(outliers), 5), 1, i+1)
            plt.imshow(data[idx, 0].cpu().numpy(), aspect='auto', origin='lower')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Anomalous Sample #{idx}')
            plt.ylabel('Frequency')
            if i == min(len(outliers), 5) - 1:
                plt.xlabel('Time')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'anomalies.png'))
        plt.close()
        
        # 保存異常樣本索引
        np.save(os.path.join(save_dir, 'anomaly_indices.npy'), outliers)
    
    return outliers

def analyze_class_distribution(labels, class_names, save_dir=None):
    """分析類別分佈"""
    print("分析類別分佈...")
    
    # 計算每個類別的樣本數
    label_counts = {}
    for i, name in enumerate(class_names):
        count = torch.sum(labels == i).item()
        label_counts[name] = count
        print(f"類別 {name}: {count} 個樣本")
    
    # 保存結果
    if save_dir:
        plt.figure(figsize=(10, 6))
        plt.bar(label_counts.keys(), label_counts.values())
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Sample Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'class_distribution.png'))
        plt.close()
        
        # 保存到CSV
        with open(os.path.join(save_dir, 'class_distribution.csv'), 'w') as f:
            f.write('Class,Count\n')
            for name, count in label_counts.items():
                f.write(f'{name},{count}\n')
    
    return label_counts

def analyze_feature_importance(data, labels, save_dir=None):
    """分析特徵重要性"""
    print("分析特徵重要性...")
    
    # 準備數據
    data_flat = data.view(data.shape[0], -1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # 使用隨機森林計算特徵重要性
    forest = RandomForestClassifier(n_estimators=100, random_state=42)
    forest.fit(data_flat, labels_np)
    
    # 特徵重要性可視化
    if save_dir:
        # 使用PCA降維
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data_flat)
        
        plt.figure(figsize=(10, 8))
        unique_labels = np.unique(labels_np)
        for label in unique_labels:
            mask = (labels_np == label)
            plt.scatter(data_pca[mask, 0], data_pca[mask, 1], label=f'Class {label}')
        
        plt.title('PCA Reduced Data Distribution')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'feature_pca.png'))
        plt.close()
    
    return {'feature_importance': forest.feature_importances_}

def cross_validate_feature_robustness(data, labels, save_dir=None):
    """交叉驗證特徵穩健性"""
    print("交叉驗證特徵穩健性...")
    
    # 準備數據
    data_flat = data.view(data.shape[0], -1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # 隨機森林交叉驗證
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(clf, data_flat, labels_np, cv=kf)
    
    print(f"交叉驗證分數: {scores}")
    print(f"平均分數: {np.mean(scores):.4f} (標準差: {np.std(scores):.4f})")
    
    # 保存結果
    if save_dir:
        with open(os.path.join(save_dir, 'cross_validation.txt'), 'w') as f:
            f.write(f"Cross-validation scores: {scores}\n")
            f.write(f"Mean score: {np.mean(scores):.4f} (Std dev: {np.std(scores):.4f})\n")
    
    return {
        'scores': scores,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores)
    }

def examine_data_leakage(data, labels, train_indices, test_indices, save_dir=None):
    """檢查數據洩漏"""
    print("檢查數據洩漏...")
    
    # 準備訓練和測試數據
    data_flat = data.view(data.shape[0], -1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    X_train = data_flat[train_indices]
    y_train = labels_np[train_indices]
    X_test = data_flat[test_indices]
    y_test = labels_np[test_indices]
    
    # 特徵相似性指標
    train_mean = np.mean(X_train, axis=0)
    test_mean = np.mean(X_test, axis=0)
    
    # 比較訓練和測試集的分佈
    train_std = np.std(X_train, axis=0)
    test_std = np.std(X_test, axis=0)
    
    # 計算相對差距
    mean_diff = np.abs(train_mean - test_mean) / (np.abs(train_mean) + 1e-10)
    std_diff = np.abs(train_std - test_std) / (np.abs(train_std) + 1e-10)
    
    # 檢測異常差異
    mean_threshold = 0.5
    std_threshold = 0.5
    
    mean_outliers = np.sum(mean_diff > mean_threshold)
    std_outliers = np.sum(std_diff > std_threshold)
    
    leakage_risk = (mean_outliers / len(mean_diff) + std_outliers / len(std_diff)) / 2
    
    print(f"平均值分佈異常特徵: {mean_outliers} / {len(mean_diff)}")
    print(f"標準差分佈異常特徵: {std_outliers} / {len(std_diff)}")
    print(f"數據洩漏風險評分: {leakage_risk:.4f}")
    
    # 保存結果
    if save_dir:
        with open(os.path.join(save_dir, 'data_leakage.txt'), 'w') as f:
            f.write(f"Mean distribution anomalies: {mean_outliers} / {len(mean_diff)}\n")
            f.write(f"Standard deviation anomalies: {std_outliers} / {len(std_diff)}\n")
            f.write(f"Data leakage risk score: {leakage_risk:.4f}\n")
        
        # 特徵差異分佈可視化
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(mean_diff, bins=50)
        plt.axvline(x=mean_threshold, color='r', linestyle='--')
        plt.title('Feature Mean Relative Differences')
        plt.xlabel('Relative Difference')
        plt.ylabel('Feature Count')
        
        plt.subplot(1, 2, 2)
        plt.hist(std_diff, bins=50)
        plt.axvline(x=std_threshold, color='r', linestyle='--')
        plt.title('Feature Std Dev Relative Differences')
        plt.xlabel('Relative Difference')
        plt.ylabel('Feature Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_distribution_diff.png'))
        plt.close()
    
    leakage_metrics = {
        'mean_outliers_ratio': mean_outliers / len(mean_diff),
        'std_outliers_ratio': std_outliers / len(std_diff),
        'leakage_risk': leakage_risk
    }
    
    return leakage_metrics

def analyze_ranking_pairs(data1, data2, targets, labels1, labels2, save_dir=None):
    """分析排序對"""
    print("分析排序對...")
    
    # 分析標籤差異與目標值的關係
    label_diffs = labels1.cpu().numpy() - labels2.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    consistent_pairs = np.sum((label_diffs > 0) & (targets_np > 0)) + np.sum((label_diffs < 0) & (targets_np < 0))
    total_pairs = len(targets_np)
    consistency_ratio = consistent_pairs / total_pairs
    
    print(f"排序對一致性: {consistent_pairs} / {total_pairs} ({consistency_ratio:.4f})")
    
    # 保存結果
    if save_dir:
        with open(os.path.join(save_dir, 'ranking_pairs.txt'), 'w') as f:
            f.write(f"Ranking pair consistency: {consistent_pairs} / {total_pairs} ({consistency_ratio:.4f})\n")
    
    return {
        'consistency_ratio': consistency_ratio,
        'consistent_pairs': consistent_pairs,
        'total_pairs': total_pairs
    }

def evaluate_ranking_consistency(targets, labels1, labels2, save_dir=None):
    """評估排序一致性"""
    print("評估排序一致性...")
    
    # 將標籤和目標轉換為NumPy數組
    targets_np = targets.cpu().numpy()
    labels1_np = labels1.cpu().numpy()
    labels2_np = labels2.cpu().numpy()
    
    # 計算標籤差與目標的關係
    label_diffs = labels1_np - labels2_np
    
    # 計算排序一致性
    expected_targets = np.sign(label_diffs)
    actual_targets = targets_np
    
    # 把為0的部分排除在外
    valid_indices = (expected_targets != 0)
    expected_filtered = expected_targets[valid_indices]
    actual_filtered = actual_targets[valid_indices]
    
    # 計算一致性比例
    matches = (expected_filtered == actual_filtered)
    consistency = np.mean(matches)
    
    print(f"排序一致性比例: {consistency:.4f}")
    
    # 保存結果
    if save_dir:
        with open(os.path.join(save_dir, 'ranking_consistency.txt'), 'w') as f:
            f.write(f"Ranking consistency ratio: {consistency:.4f}\n")
    
    return {
        'consistency': consistency
    }

def visualize_ranking_embeddings(data, labels, save_dir=None):
    """可視化排序嵌入"""
    print("可視化排序嵌入...")
    
    # 使用PCA降維以便可視化
    data_flat = data.view(data.shape[0], -1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # 執行PCA
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_flat)
    
    # 保存可視化結果
    if save_dir:
        plt.figure(figsize=(10, 8))
        
        # 繪製每個類別的樣本
        unique_labels = np.unique(labels_np)
        for label in unique_labels:
            mask = (labels_np == label)
            plt.scatter(data_pca[mask, 0], data_pca[mask, 1], label=f'Class {label}')
        
        plt.title('PCA Reduced Data Embeddings')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'embedding_visualization.png'))
        plt.close()
    
    return {
        'pca_explained_variance': pca.explained_variance_ratio_
    } 