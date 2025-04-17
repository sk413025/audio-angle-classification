"""
數據集分析工具：用於檢查數據集的品質和特性
功能：
- 分析數據集的類別分佈
- 評估特徵的重要性和貢獻度
- 交叉驗證特徵的穩健性
- 檢查數據洩漏問題
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import os

def analyze_class_distribution(labels, class_names=None, save_dir=None):
    """
    分析數據集的類別分佈情況
    
    參數:
        labels: 類別標籤數組
        class_names: 類別名稱列表 (可選)
        save_dir: 保存結果的目錄路徑
    
    返回:
        class_stats: 包含類別統計資訊的字典
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # 計算每個類別的樣本數量
    unique_labels, class_counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)
    
    # 構建類別統計字典
    class_stats = {
        'unique_labels': unique_labels,
        'class_counts': class_counts,
        'class_percentages': (class_counts / total_samples) * 100,
        'total_samples': total_samples,
        'n_classes': len(unique_labels),
        'imbalance_ratio': np.max(class_counts) / np.min(class_counts) if len(class_counts) > 0 else 0
    }
    
    # 輸出統計摘要
    print(f"\n類別分佈分析摘要:")
    print(f"總樣本數: {total_samples}")
    print(f"類別數量: {class_stats['n_classes']}")
    print(f"不平衡比率: {class_stats['imbalance_ratio']:.2f}:1")
    print("\n類別分佈:")
    
    for i, label in enumerate(unique_labels):
        label_name = class_names[label] if class_names is not None else f"Class {label}"
        print(f"  {label_name}: {class_counts[i]} samples ({class_stats['class_percentages'][i]:.2f}%)")
    
    # 繪製類別分佈圖
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        
        # 準備類別名稱
        if class_names is not None:
            plot_labels = [class_names[label] for label in unique_labels]
        else:
            plot_labels = [f"Class {label}" for label in unique_labels]
        
        # 繪製類別分佈條形圖
        plt.subplot(1, 2, 1)
        sns.barplot(x=plot_labels, y=class_counts)
        plt.title('數據集類別分佈')
        plt.xlabel('類別')
        plt.ylabel('樣本數量')
        plt.xticks(rotation=45)
        
        # 繪製類別分佈餅圖
        plt.subplot(1, 2, 2)
        plt.pie(class_counts, labels=plot_labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('類別分佈百分比')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'class_distribution.png'))
        plt.close()
    
    return class_stats

def analyze_feature_importance(spectrograms, labels, n_features=10, save_dir=None):
    """
    評估頻譜圖特徵的重要性
    
    參數:
        spectrograms: 頻譜圖張量 [batch, channels, freq, time]
        labels: 類別標籤數組
        n_features: 報告的頂部特徵數量
        save_dir: 保存結果的目錄路徑
    
    返回:
        importance_metrics: 包含特徵重要性評估的字典
    """
    if isinstance(spectrograms, torch.Tensor):
        spectrograms = spectrograms.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # 將頻譜圖調整為2D格式 [samples, features]
    batch_size = spectrograms.shape[0]
    orig_shape = spectrograms.shape
    flattened = spectrograms.reshape(batch_size, -1)
    
    # 對特徵進行降維以加快處理速度
    n_components = min(100, flattened.shape[1])
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(flattened)
    
    # 隨機森林特徵重要性
    forest = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    forest.fit(reduced_features, labels)
    rf_importances = forest.feature_importances_
    
    # 互信息特徵重要性
    mi_importances = mutual_info_classif(reduced_features, labels, random_state=42)
    
    # 找出最重要的特徵
    top_rf_indices = np.argsort(rf_importances)[-n_features:][::-1]
    top_mi_indices = np.argsort(mi_importances)[-n_features:][::-1]
    
    # 構建返回字典
    importance_metrics = {
        'pca_components': pca.components_,
        'pca_explained_variance': pca.explained_variance_ratio_,
        'rf_importances': rf_importances,
        'mi_importances': mi_importances,
        'top_rf_indices': top_rf_indices,
        'top_mi_indices': top_mi_indices,
        'top_rf_values': rf_importances[top_rf_indices],
        'top_mi_values': mi_importances[top_mi_indices]
    }
    
    # 輸出統計摘要
    print(f"\n特徵重要性分析摘要:")
    print(f"PCA降維後特徵數: {n_components}")
    print(f"PCA解釋方差: {sum(pca.explained_variance_ratio_)*100:.2f}%")
    
    print(f"\n隨機森林特徵重要性 (前{n_features}個):")
    for i, idx in enumerate(top_rf_indices):
        print(f"  特徵 {idx}: {rf_importances[idx]:.4f}")
    
    print(f"\n互信息特徵重要性 (前{n_features}個):")
    for i, idx in enumerate(top_mi_indices):
        print(f"  特徵 {idx}: {mi_importances[idx]:.4f}")
    
    # 繪製特徵重要性圖
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(15, 10))
        
        # PCA解釋方差
        plt.subplot(2, 2, 1)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('成分數量')
        plt.ylabel('累積解釋方差')
        plt.title('PCA累積解釋方差')
        plt.grid(True)
        
        # 隨機森林特徵重要性
        plt.subplot(2, 2, 2)
        plt.bar(range(len(top_rf_indices)), importance_metrics['top_rf_values'])
        plt.xlabel('特徵索引')
        plt.ylabel('重要性')
        plt.title('隨機森林特徵重要性 (前N個)')
        plt.xticks(range(len(top_rf_indices)), [f"{idx}" for idx in top_rf_indices], rotation=45)
        
        # 互信息特徵重要性
        plt.subplot(2, 2, 3)
        plt.bar(range(len(top_mi_indices)), importance_metrics['top_mi_values'])
        plt.xlabel('特徵索引')
        plt.ylabel('互信息')
        plt.title('互信息特徵重要性 (前N個)')
        plt.xticks(range(len(top_mi_indices)), [f"{idx}" for idx in top_mi_indices], rotation=45)
        
        # 重要特徵的相關性
        try:
            # 所有重要特徵的索引
            important_indices = np.unique(np.concatenate([top_rf_indices, top_mi_indices]))
            important_features = reduced_features[:, important_indices]
            
            # 計算重要特徵間的相關性
            corr = np.corrcoef(important_features.T)
            
            # 繪製相關性熱圖
            plt.subplot(2, 2, 4)
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('重要特徵間的相關性')
            plt.tight_layout()
        except:
            print("無法計算特徵相關性")
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_importance.png'))
        plt.close()
        
        # 繪製PCA成分圖
        try:
            plt.figure(figsize=(12, 10))
            
            # 前兩個PCA成分的散點圖
            plt.subplot(2, 2, 1)
            unique_labels = np.unique(labels)
            for label in unique_labels:
                idx = np.where(labels == label)[0]
                plt.scatter(reduced_features[idx, 0], reduced_features[idx, 1], label=f'Class {label}', alpha=0.7)
            plt.xlabel('PCA成分 1')
            plt.ylabel('PCA成分 2')
            plt.title('PCA投影 (成分 1 vs 成分 2)')
            plt.legend()
            
            # 前兩個主要成分的分佈
            plt.subplot(2, 2, 2)
            for label in unique_labels:
                idx = np.where(labels == label)[0]
                sns.kdeplot(reduced_features[idx, 0], label=f'Class {label}', fill=True, alpha=0.3)
            plt.xlabel('PCA成分 1')
            plt.ylabel('密度')
            plt.title('PCA成分 1 分佈 (按類別)')
            plt.legend()
            
            # 重要PCA成分的可視化
            plt.subplot(2, 2, 3)
            plt.imshow(pca.components_[:5], aspect='auto', cmap='viridis')
            plt.colorbar()
            plt.xlabel('原始特徵索引')
            plt.ylabel('PCA成分')
            plt.title('前5個PCA成分')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'pca_components.png'))
            plt.close()
        except Exception as e:
            print(f"繪製PCA成分時出錯: {str(e)}")
    
    return importance_metrics

def cross_validate_feature_robustness(spectrograms, labels, n_splits=5, save_dir=None):
    """
    交叉驗證特徵的穩健性
    
    參數:
        spectrograms: 頻譜圖張量 [batch, channels, freq, time]
        labels: 類別標籤數組
        n_splits: K-fold交叉驗證的折數
        save_dir: 保存結果的目錄路徑
    
    返回:
        cv_results: 交叉驗證結果字典
    """
    if isinstance(spectrograms, torch.Tensor):
        spectrograms = spectrograms.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # 將頻譜圖調整為2D格式 [samples, features]
    batch_size = spectrograms.shape[0]
    flattened = spectrograms.reshape(batch_size, -1)
    
    # 對特徵進行降維以加快處理速度
    n_components = min(100, flattened.shape[1])
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(flattened)
    
    # 使用5折交叉驗證對隨機森林進行評估
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    # 計算不同性能指標的交叉驗證分數
    accuracy_scores = cross_val_score(clf, reduced_features, labels, cv=cv, scoring='accuracy')
    f1_scores = cross_val_score(clf, reduced_features, labels, cv=cv, scoring='f1_weighted')
    precision_scores = cross_val_score(clf, reduced_features, labels, cv=cv, scoring='precision_weighted')
    recall_scores = cross_val_score(clf, reduced_features, labels, cv=cv, scoring='recall_weighted')
    
    # 在每一折上單獨訓練模型並獲取重要性
    importances_per_fold = []
    for train_idx, test_idx in cv.split(reduced_features):
        X_train, X_test = reduced_features[train_idx], reduced_features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        clf.fit(X_train, y_train)
        importances_per_fold.append(clf.feature_importances_)
    
    importances_array = np.array(importances_per_fold)
    mean_importances = np.mean(importances_array, axis=0)
    std_importances = np.std(importances_array, axis=0)
    
    # 構建返回字典
    cv_results = {
        'accuracy_scores': accuracy_scores,
        'f1_scores': f1_scores,
        'precision_scores': precision_scores,
        'recall_scores': recall_scores,
        'mean_accuracy': np.mean(accuracy_scores),
        'std_accuracy': np.std(accuracy_scores),
        'mean_f1': np.mean(f1_scores),
        'std_f1': np.std(f1_scores),
        'importances_per_fold': importances_array,
        'mean_importances': mean_importances,
        'std_importances': std_importances,
        'top_stable_features': np.argsort(mean_importances / (std_importances + 1e-10))[-10:][::-1]
    }
    
    # 輸出統計摘要
    print(f"\n特徵穩健性交叉驗證摘要 ({n_splits}-fold):")
    print(f"精確度: {cv_results['mean_accuracy']:.4f} (±{cv_results['std_accuracy']:.4f})")
    print(f"F1分數: {cv_results['mean_f1']:.4f} (±{cv_results['std_f1']:.4f})")
    print(f"精準率: {np.mean(precision_scores):.4f} (±{np.std(precision_scores):.4f})")
    print(f"召回率: {np.mean(recall_scores):.4f} (±{np.std(recall_scores):.4f})")
    
    # 繪製交叉驗證結果圖
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(15, 10))
        
        # 交叉驗證性能指標
        plt.subplot(2, 2, 1)
        metrics = ['Accuracy', 'F1', 'Precision', 'Recall']
        mean_scores = [
            cv_results['mean_accuracy'], 
            cv_results['mean_f1'],
            np.mean(precision_scores),
            np.mean(recall_scores)
        ]
        std_scores = [
            cv_results['std_accuracy'], 
            cv_results['std_f1'],
            np.std(precision_scores),
            np.std(recall_scores)
        ]
        
        plt.bar(metrics, mean_scores, yerr=std_scores, capsize=10)
        plt.ylim([0, 1.1])
        plt.title('交叉驗證性能指標')
        plt.ylabel('分數')
        
        # 不同折上的精確度
        plt.subplot(2, 2, 2)
        plt.plot(range(1, n_splits+1), accuracy_scores, 'o-')
        plt.axhline(y=cv_results['mean_accuracy'], color='r', linestyle='--')
        plt.title('不同折上的精確度')
        plt.xlabel('折數')
        plt.ylabel('精確度')
        plt.xticks(range(1, n_splits+1))
        plt.grid(True)
        
        # 前10個特徵的重要性方差
        plt.subplot(2, 2, 3)
        top_features = np.argsort(mean_importances)[-10:][::-1]
        plt.errorbar(range(len(top_features)), 
                    mean_importances[top_features], 
                    yerr=std_importances[top_features], 
                    fmt='o', 
                    capsize=5)
        plt.title('前10個特徵重要性及變異')
        plt.xlabel('特徵索引')
        plt.ylabel('重要性')
        plt.xticks(range(len(top_features)), [f"{idx}" for idx in top_features], rotation=45)
        plt.grid(True)
        
        # 不同折上重要性的熱圖
        plt.subplot(2, 2, 4)
        # 所有折上的前10個特徵重要性
        fold_importances = importances_array[:, top_features]
        sns.heatmap(fold_importances, cmap='viridis')
        plt.title('不同折上的特徵重要性')
        plt.xlabel('特徵索引')
        plt.ylabel('折數')
        plt.yticks(range(n_splits), [f"Fold {i+1}" for i in range(n_splits)])
        plt.xticks(range(len(top_features)), [f"{idx}" for idx in top_features], rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_robustness.png'))
        plt.close()
        
    return cv_results

def examine_data_leakage(spectrograms, labels, train_indices=None, test_indices=None, save_dir=None):
    """
    檢查訓練集和測試集之間的潛在數據洩漏
    
    參數:
        spectrograms: 頻譜圖張量 [batch, channels, freq, time]
        labels: 類別標籤數組
        train_indices: 訓練集樣本的索引
        test_indices: 測試集樣本的索引
        save_dir: 保存結果的目錄路徑
    
    返回:
        leakage_metrics: 包含洩漏評估的字典
    """
    if isinstance(spectrograms, torch.Tensor):
        spectrograms = spectrograms.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # 如果未提供訓練和測試索引，則使用基本的分割
    if train_indices is None or test_indices is None:
        n_samples = spectrograms.shape[0]
        indices = np.random.permutation(n_samples)
        split = int(n_samples * 0.8)  # 80% 訓練, 20% 測試
        train_indices = indices[:split]
        test_indices = indices[split:]
    
    # 將頻譜圖調整為2D格式 [samples, features]
    batch_size = spectrograms.shape[0]
    flattened = spectrograms.reshape(batch_size, -1)
    
    # 提取訓練集和測試集
    X_train = flattened[train_indices]
    y_train = labels[train_indices]
    X_test = flattened[test_indices]
    y_test = labels[test_indices]
    
    # 對特徵進行降維以加快處理速度
    n_components = min(50, flattened.shape[1])
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # 計算與最近鄰的距離 (訓練到測試樣本)
    from sklearn.neighbors import NearestNeighbors
    
    # 為每個測試樣本找到最近的訓練樣本
    nbrs = NearestNeighbors(n_neighbors=1).fit(X_train_pca)
    distances, indices = nbrs.kneighbors(X_test_pca)
    distances = distances.flatten()
    
    # 找到最相似的訓練-測試樣本對
    most_similar_idx = np.argsort(distances)[:10]  # 前10個最相似的對
    most_similar_test_idx = [test_indices[idx] for idx in most_similar_idx]
    most_similar_train_idx = [train_indices[indices[idx][0]] for idx in most_similar_idx]
    most_similar_distances = distances[most_similar_idx]
    
    # 計算訓練集內部距離分佈
    n_train_samples = min(500, len(train_indices))  # 限制樣本數以加快速度
    train_sample_indices = np.random.choice(len(train_indices), n_train_samples, replace=False)
    X_train_sample = X_train_pca[train_sample_indices]
    
    nbrs_train = NearestNeighbors(n_neighbors=2).fit(X_train_sample)  # 2個鄰居是因為樣本本身是最近的
    train_distances, _ = nbrs_train.kneighbors(X_train_sample)
    train_distances = train_distances[:, 1]  # 忽略自身距離，只取第二近的
    
    # 檢查測試集距離是否顯著小於訓練集內部距離
    suspicious_threshold = np.percentile(train_distances, 10)  # 10% 分位數
    suspicious_pairs = distances < suspicious_threshold
    num_suspicious = np.sum(suspicious_pairs)
    
    # 構建返回字典
    leakage_metrics = {
        'test_to_train_distances': distances,
        'train_internal_distances': train_distances,
        'suspicious_threshold': suspicious_threshold,
        'num_suspicious_pairs': num_suspicious,
        'suspicious_percentage': (num_suspicious / len(test_indices)) * 100,
        'most_similar_test_idx': most_similar_test_idx,
        'most_similar_train_idx': most_similar_train_idx,
        'most_similar_distances': most_similar_distances
    }
    
    # 輸出統計摘要
    print(f"\n數據洩漏檢查摘要:")
    print(f"訓練集樣本數: {len(train_indices)}")
    print(f"測試集樣本數: {len(test_indices)}")
    print(f"訓練集內部距離中位數: {np.median(train_distances):.6f}")
    print(f"測試到訓練集距離中位數: {np.median(distances):.6f}")
    print(f"可疑樣本對數量: {num_suspicious} ({leakage_metrics['suspicious_percentage']:.2f}%)")
    
    if num_suspicious > 0:
        print("\n前5個最相似的測試-訓練樣本對:")
        for i in range(min(5, len(most_similar_test_idx))):
            test_idx = most_similar_test_idx[i]
            train_idx = most_similar_train_idx[i]
            dist = most_similar_distances[i]
            print(f"  測試樣本 {test_idx} 與訓練樣本 {train_idx}, 距離 = {dist:.6f}")
    
    # 繪製數據洩漏檢查圖
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(15, 10))
        
        # 訓練集內部距離和測試到訓練集距離的分佈
        plt.subplot(2, 2, 1)
        sns.histplot(train_distances, kde=True, color='blue', label='訓練集內部距離')
        sns.histplot(distances, kde=True, color='red', label='測試到訓練集距離')
        plt.axvline(x=suspicious_threshold, color='green', linestyle='--', label='可疑閾值')
        plt.title('距離分佈比較')
        plt.xlabel('距離')
        plt.ylabel('頻率')
        plt.legend()
        
        # 測試和訓練樣本在PCA空間中的分佈
        plt.subplot(2, 2, 2)
        plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], color='blue', alpha=0.5, label='訓練樣本')
        plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], color='red', alpha=0.5, label='測試樣本')
        
        # 標記可疑樣本對
        if num_suspicious > 0:
            suspicious_test_idx = np.where(suspicious_pairs)[0]
            suspicious_test_points = X_test_pca[suspicious_test_idx]
            plt.scatter(suspicious_test_points[:, 0], suspicious_test_points[:, 1], 
                       color='green', marker='x', s=100, label='可疑測試樣本')
        
        plt.title('PCA空間中的樣本分佈')
        plt.xlabel('PCA成分 1')
        plt.ylabel('PCA成分 2')
        plt.legend()
        
        # 最相似測試-訓練對的距離
        plt.subplot(2, 2, 3)
        plt.bar(range(min(10, len(most_similar_distances))), most_similar_distances[:10])
        plt.axhline(y=suspicious_threshold, color='r', linestyle='--', label='可疑閾值')
        plt.title('最相似的10個測試-訓練對的距離')
        plt.xlabel('對索引')
        plt.ylabel('距離')
        plt.legend()
        
        # 繪製距離矩陣熱圖
        plt.subplot(2, 2, 4)
        # 從測試集和訓練集中隨機選擇樣本以繪製距離矩陣
        n_samples = min(20, min(len(test_indices), len(train_indices)))
        test_sample_idx = np.random.choice(len(test_indices), n_samples, replace=False)
        train_sample_idx = np.random.choice(len(train_indices), n_samples, replace=False)
        
        X_test_sample = X_test_pca[test_sample_idx]
        X_train_sample = X_train_pca[train_sample_idx]
        
        from sklearn.metrics.pairwise import euclidean_distances
        dist_matrix = euclidean_distances(X_test_sample, X_train_sample)
        
        sns.heatmap(dist_matrix, cmap='viridis')
        plt.title('測試-訓練樣本距離矩陣')
        plt.xlabel('訓練樣本')
        plt.ylabel('測試樣本')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'data_leakage_check.png'))
        plt.close()
        
    return leakage_metrics 