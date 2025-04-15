"""
排序診斷工具：用於分析排序對數據的品質和特性
功能：
- 分析排序對的品質和分佈
- 評估排序一致性與衝突
- 視覺化排序嵌入空間
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from itertools import combinations
import os

def analyze_ranking_pairs(data1, data2, targets, labels1=None, labels2=None, save_dir=None):
    """
    分析排序對數據的品質和分佈
    
    參數:
        data1: 排序對中的第一組數據 [batch, ...]
        data2: 排序對中的第二組數據 [batch, ...]
        targets: 排序目標 (1表示data1優於data2, -1表示data2優於data1)
        labels1: 第一組數據的類別標籤 (可選)
        labels2: 第二組數據的類別標籤 (可選)
        save_dir: 保存結果的目錄路徑
    
    返回:
        pair_stats: 包含排序對統計資訊的字典
    """
    if isinstance(data1, torch.Tensor):
        data1 = data1.detach().cpu().numpy()
    if isinstance(data2, torch.Tensor):
        data2 = data2.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(labels1, torch.Tensor):
        labels1 = labels1.detach().cpu().numpy()
    if isinstance(labels2, torch.Tensor):
        labels2 = labels2.detach().cpu().numpy()
    
    n_pairs = len(targets)
    
    # 計算基本統計量
    positive_pairs = np.sum(targets == 1)
    negative_pairs = np.sum(targets == -1)
    neutral_pairs = n_pairs - positive_pairs - negative_pairs
    
    # 如果有標籤信息，分析類別間的排序關係
    class_transitions = None
    if labels1 is not None and labels2 is not None:
        unique_labels = np.unique(np.concatenate([labels1, labels2]))
        n_classes = len(unique_labels)
        class_transitions = np.zeros((n_classes, n_classes))
        
        for i in range(n_pairs):
            if targets[i] == 1:  # data1優先
                class_transitions[labels1[i], labels2[i]] += 1
            elif targets[i] == -1:  # data2優先
                class_transitions[labels2[i], labels1[i]] += 1
    
    # 構建返回字典
    pair_stats = {
        'n_pairs': n_pairs,
        'positive_pairs': positive_pairs,
        'negative_pairs': negative_pairs,
        'neutral_pairs': neutral_pairs,
        'positive_ratio': positive_pairs / n_pairs if n_pairs > 0 else 0,
        'negative_ratio': negative_pairs / n_pairs if n_pairs > 0 else 0,
        'neutral_ratio': neutral_pairs / n_pairs if n_pairs > 0 else 0,
        'class_transitions': class_transitions
    }
    
    # 輸出統計摘要
    print(f"\n排序對分析摘要:")
    print(f"總排序對數量: {n_pairs}")
    print(f"正向排序對 (data1優先): {positive_pairs} ({pair_stats['positive_ratio']*100:.2f}%)")
    print(f"負向排序對 (data2優先): {negative_pairs} ({pair_stats['negative_ratio']*100:.2f}%)")
    
    if neutral_pairs > 0:
        print(f"中性排序對 (無偏好): {neutral_pairs} ({pair_stats['neutral_ratio']*100:.2f}%)")
    
    # 計算數據集中每個類別的平均排序地位
    if labels1 is not None and labels2 is not None:
        class_ranks = {}
        for label in unique_labels:
            wins = 0
            losses = 0
            
            # 當label出現在data1中的勝負
            mask1 = labels1 == label
            wins += np.sum((targets[mask1] == 1))
            losses += np.sum((targets[mask1] == -1))
            
            # 當label出現在data2中的勝負
            mask2 = labels2 == label
            wins += np.sum((targets[mask2] == -1))
            losses += np.sum((targets[mask2] == 1))
            
            # 計算勝率
            total = wins + losses
            if total > 0:
                win_rate = wins / total
            else:
                win_rate = 0
                
            class_ranks[label] = win_rate
        
        pair_stats['class_ranks'] = class_ranks
        
        print("\n類別排序地位:")
        for label, rank in sorted(class_ranks.items(), key=lambda x: x[1], reverse=True):
            print(f"  類別 {label}: 勝率 = {rank:.4f}")
    
    # 繪製排序對分析圖
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(15, 10))
        
        # 排序對分佈餅圖
        plt.subplot(2, 2, 1)
        labels = ['Data1優先', 'Data2優先']
        sizes = [positive_pairs, negative_pairs]
        if neutral_pairs > 0:
            labels.append('無偏好')
            sizes.append(neutral_pairs)
        
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('排序對分佈')
        
        if class_transitions is not None and len(unique_labels) > 1:
            # 類別間轉移矩陣熱圖
            plt.subplot(2, 2, 2)
            sns.heatmap(class_transitions, annot=True, fmt='.0f', cmap='viridis')
            plt.title('類別間優先關係矩陣')
            plt.xlabel('劣勢類別')
            plt.ylabel('優勢類別')
            
            # 類別排序地位條形圖
            plt.subplot(2, 2, 3)
            sorted_ranks = sorted(class_ranks.items(), key=lambda x: x[1], reverse=True)
            labels, values = zip(*sorted_ranks)
            plt.bar(range(len(labels)), values)
            plt.xticks(range(len(labels)), [f"類別 {l}" for l in labels])
            plt.ylim([0, 1])
            plt.title('類別排序地位 (勝率)')
            plt.xlabel('類別')
            plt.ylabel('勝率')
            
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'ranking_pairs_analysis.png'))
        plt.close()
    
    return pair_stats

def evaluate_ranking_consistency(targets, labels1, labels2, save_dir=None):
    """
    評估排序對數據的一致性，檢測衝突
    
    參數:
        targets: 排序目標 (1表示data1優於data2, -1表示data2優於data1)
        labels1: 第一組數據的類別標籤
        labels2: 第二組數據的類別標籤
        save_dir: 保存結果的目錄路徑
    
    返回:
        consistency_metrics: 包含一致性評估的字典
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(labels1, torch.Tensor):
        labels1 = labels1.detach().cpu().numpy()
    if isinstance(labels2, torch.Tensor):
        labels2 = labels2.detach().cpu().numpy()
    
    n_pairs = len(targets)
    
    # 創建類別間的排序關係矩陣
    unique_labels = np.unique(np.concatenate([labels1, labels2]))
    n_classes = len(unique_labels)
    preference_matrix = np.zeros((n_classes, n_classes))
    count_matrix = np.zeros((n_classes, n_classes))
    
    # 填充排序關係矩陣
    for i in range(n_pairs):
        l1, l2 = labels1[i], labels2[i]
        if targets[i] == 1:  # data1優先
            preference_matrix[l1, l2] += 1
        elif targets[i] == -1:  # data2優先
            preference_matrix[l2, l1] += 1
        
        count_matrix[l1, l2] += 1
        count_matrix[l2, l1] += 1
    
    # 轉換為機率
    with np.errstate(divide='ignore', invalid='ignore'):
        probability_matrix = np.where(count_matrix > 0, preference_matrix / count_matrix, 0)
    
    # 檢測排序衝突
    conflicts = []
    for a, b in combinations(range(n_classes), 2):
        if preference_matrix[a, b] > 0 and preference_matrix[b, a] > 0:
            conflicts.append((a, b, preference_matrix[a, b], preference_matrix[b, a]))
    
    # 計算一致性指標
    transitivity_violations = 0
    for a, b, c in combinations(range(n_classes), 3):
        # 如果 a > b 且 b > c，則應該 a > c
        if (probability_matrix[a, b] > 0.5 and probability_matrix[b, c] > 0.5 and 
            probability_matrix[a, c] < 0.5):
            transitivity_violations += 1
    
    # 構建返回字典
    consistency_metrics = {
        'preference_matrix': preference_matrix,
        'probability_matrix': probability_matrix,
        'conflicts': conflicts,
        'transitivity_violations': transitivity_violations,
        'n_conflicts': len(conflicts),
        'conflict_ratio': len(conflicts) / (n_classes * (n_classes - 1) / 2) if n_classes > 1 else 0,
        'transitivity_ratio': transitivity_violations / (n_classes * (n_classes - 1) * (n_classes - 2) / 6) if n_classes > 2 else 0
    }
    
    # 輸出統計摘要
    print(f"\n排序一致性分析摘要:")
    print(f"類別總數: {n_classes}")
    print(f"排序衝突對數: {len(conflicts)} (佔比 {consistency_metrics['conflict_ratio']*100:.2f}%)")
    print(f"遞移性違規數: {transitivity_violations} (佔比 {consistency_metrics['transitivity_ratio']*100:.2f}%)")
    
    if len(conflicts) > 0:
        print("\n排序衝突詳情:")
        for a, b, ab_count, ba_count in conflicts[:5]:  # 只顯示前5個
            total = ab_count + ba_count
            print(f"  類別 {a} vs 類別 {b}: {ab_count}/{total} vs {ba_count}/{total}")
        
        if len(conflicts) > 5:
            print(f"  ... 以及 {len(conflicts)-5} 個其他衝突")
    
    # 繪製一致性分析圖
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(15, 10))
        
        # 類別間優先概率熱圖
        plt.subplot(2, 2, 1)
        sns.heatmap(probability_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=0, vmax=1)
        plt.title('類別間優先機率矩陣')
        plt.xlabel('類別 j')
        plt.ylabel('類別 i')
        plt.title('P(i優於j)')
        
        # 衝突矩陣
        conflict_matrix = np.zeros((n_classes, n_classes))
        for a, b, ab_count, ba_count in conflicts:
            total = ab_count + ba_count
            # 衝突程度 (0.5表示完全均衡, 1表示無衝突)
            conflict_strength = max(ab_count, ba_count) / total
            conflict_matrix[a, b] = conflict_strength
            conflict_matrix[b, a] = conflict_strength
        
        plt.subplot(2, 2, 2)
        sns.heatmap(conflict_matrix, annot=True, fmt='.2f', cmap='YlOrRd', vmin=0.5, vmax=1)
        plt.title('排序衝突強度矩陣')
        plt.xlabel('類別')
        plt.ylabel('類別')
        
        # 排序衝突網絡
        if len(conflicts) > 0:
            plt.subplot(2, 2, 3)
            
            # 創建一個簡單的有向圖
            from matplotlib.lines import Line2D
            
            # 為各節點計算位置 (圓形分佈)
            pos = {}
            for i, label in enumerate(unique_labels):
                angle = 2 * np.pi * i / n_classes
                pos[label] = (np.cos(angle), np.sin(angle))
            
            # 先繪製所有節點
            for label, (x, y) in pos.items():
                plt.scatter(x, y, s=300, color='skyblue')
                plt.text(x, y, f'{label}', horizontalalignment='center', verticalalignment='center')
            
            # 然後繪製有向邊
            for a, b, ab_count, ba_count in conflicts:
                x1, y1 = pos[a]
                x2, y2 = pos[b]
                
                # 計算邊的權重
                total = ab_count + ba_count
                a_to_b_weight = ab_count / total
                b_to_a_weight = ba_count / total
                
                # 只畫主導方向的邊
                if a_to_b_weight > b_to_a_weight:
                    plt.arrow(x1, y1, (x2-x1)*0.8, (y2-y1)*0.8, 
                             head_width=0.05, head_length=0.1, fc='blue', ec='blue', 
                             alpha=a_to_b_weight)
                else:
                    plt.arrow(x2, y2, (x1-x2)*0.8, (y1-y2)*0.8, 
                             head_width=0.05, head_length=0.1, fc='red', ec='red', 
                             alpha=b_to_a_weight)
            
            plt.title('排序關係圖 (箭頭指向優勢類別)')
            plt.xlim(-1.5, 1.5)
            plt.ylim(-1.5, 1.5)
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'ranking_consistency.png'))
        plt.close()
    
    return consistency_metrics

def visualize_ranking_embeddings(data, labels=None, model=None, save_dir=None):
    """
    視覺化排序數據的嵌入空間
    
    參數:
        data: 要嵌入的數據 [batch, ...]
        labels: 數據的類別標籤 (可選)
        model: 排序模型 (可選)，用於獲取特徵嵌入
        save_dir: 保存結果的目錄路徑
    
    返回:
        embeddings: 數據在低維空間的嵌入
    """
    if isinstance(data, torch.Tensor):
        data_tensor = data
        data = data.detach().cpu().numpy()
    else:
        data_tensor = torch.tensor(data)
    
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # 如果提供了模型，使用模型來提取特徵
    if model is not None and hasattr(model, 'backbone'):
        # 確保模型處於評估模式
        model.eval()
        with torch.no_grad():
            try:
                # 取得設備
                device = next(model.parameters()).device
                # 將數據移動到與模型相同的設備
                features = model.backbone(data_tensor.to(device))
                if isinstance(features, torch.Tensor):
                    features = features.detach().cpu().numpy()
            except Exception as e:
                print(f"使用模型提取特徵時出錯: {str(e)}")
                features = None
    else:
        features = None
    
    # 將數據展平為2D格式
    batch_size = data.shape[0]
    if features is None:
        flattened = data.reshape(batch_size, -1)
    else:
        flattened = features
    
    # 使用PCA進行降維
    pca = PCA(n_components=min(50, flattened.shape[1]))
    pca_result = pca.fit_transform(flattened)
    
    # 使用t-SNE進一步降維到2D
    tsne = TSNE(n_components=2, perplexity=min(30, batch_size-1), n_iter=1000, random_state=42)
    tsne_result = tsne.fit_transform(pca_result)
    
    # 構建返回字典
    embeddings = {
        'pca_embeddings': pca_result,
        'tsne_embeddings': tsne_result,
        'pca_explained_variance': pca.explained_variance_ratio_
    }
    
    # 輸出統計摘要
    print(f"\n排序嵌入可視化摘要:")
    print(f"數據樣本數: {batch_size}")
    print(f"PCA解釋方差: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
    
    # 繪製嵌入可視化圖
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(15, 10))
        
        # t-SNE嵌入散點圖
        plt.subplot(2, 2, 1)
        if labels is not None:
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                plt.scatter(tsne_result[mask, 0], tsne_result[mask, 1], label=f'類別 {label}', alpha=0.7)
            plt.legend()
        else:
            plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.7)
        
        plt.title('t-SNE嵌入')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        
        # PCA前兩個成分散點圖
        plt.subplot(2, 2, 2)
        if labels is not None:
            for label in unique_labels:
                mask = labels == label
                plt.scatter(pca_result[mask, 0], pca_result[mask, 1], label=f'類別 {label}', alpha=0.7)
            plt.legend()
        else:
            plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
        
        plt.title('PCA嵌入 (前兩個成分)')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        
        # PCA解釋方差
        plt.subplot(2, 2, 3)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('成分數量')
        plt.ylabel('累積解釋方差')
        plt.title('PCA解釋方差累積曲線')
        plt.grid(True)
        
        # 如果有標籤，分析類別分離度
        if labels is not None:
            from sklearn.metrics import silhouette_score
            
            try:
                # 計算輪廓分數
                silhouette_pca = silhouette_score(pca_result, labels)
                silhouette_tsne = silhouette_score(tsne_result, labels)
                
                plt.subplot(2, 2, 4)
                plt.bar(['PCA', 't-SNE'], [silhouette_pca, silhouette_tsne])
                plt.ylim([-1, 1])
                plt.title('嵌入空間中的類別分離度 (輪廓分數)')
                plt.ylabel('輪廓分數')
                
                print(f"PCA嵌入輪廓分數: {silhouette_pca:.4f}")
                print(f"t-SNE嵌入輪廓分數: {silhouette_tsne:.4f}")
                
                embeddings['silhouette_pca'] = silhouette_pca
                embeddings['silhouette_tsne'] = silhouette_tsne
            except Exception as e:
                print(f"計算輪廓分數時出錯: {str(e)}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'ranking_embeddings.png'))
        plt.close()
    
    return embeddings 