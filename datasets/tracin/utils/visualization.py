"""
TracIn 視覺化工具

這個模組提供視覺化 TracIn 影響力分數和相關分析結果的工具。
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import json
from typing import Dict, List, Union, Tuple, Optional, Any
import seaborn as sns
from pathlib import Path


def plot_influence_distribution(
    influence_scores: Dict[str, float],
    save_path: Optional[str] = None,
    title: str = "Distribution of Influence Scores",
    figsize: Tuple[int, int] = (10, 6),
    bins: int = 30
):
    """
    繪製影響力分數的分佈直方圖。
    
    Args:
        influence_scores: 影響力分數字典，鍵為樣本 ID，值為影響力分數
        save_path: 保存圖表的路徑（可選）
        title: 圖表標題
        figsize: 圖表尺寸
        bins: 直方圖的箱數
    """
    plt.figure(figsize=figsize)
    
    # 提取影響力分數
    scores = list(influence_scores.values())
    
    # 繪製直方圖
    plt.hist(scores, bins=bins, alpha=0.75, color='skyblue', edgecolor='black')
    
    # 添加垂直線表示平均值和中位數
    mean_score = np.mean(scores)
    median_score = np.median(scores)
    plt.axvline(mean_score, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_score:.2f}')
    plt.axvline(median_score, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_score:.2f}')
    
    # 添加圖表標題和標籤
    plt.title(title)
    plt.xlabel('Influence Score')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存圖表
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Influence distribution plot saved to: {save_path}")
    
    plt.close()


def plot_harmful_samples(
    harmful_samples: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    title: str = "Harmful Samples by Negative Influence",
    figsize: Tuple[int, int] = (12, 8),
    max_samples: int = 20
):
    """
    繪製有害樣本的負面影響力條形圖。
    
    Args:
        harmful_samples: 有害樣本列表，每個項目為一個字典，包含 'sample_id' 和 'average_influence' 字段
        save_path: 保存圖表的路徑（可選）
        title: 圖表標題
        figsize: 圖表尺寸
        max_samples: 要顯示的最大樣本數
    """
    # 限制樣本數量
    samples_to_plot = harmful_samples[:max_samples]
    
    plt.figure(figsize=figsize)
    
    # 提取樣本 ID 和平均影響力
    sample_ids = [item['sample_id'] for item in samples_to_plot]
    avg_influences = [item['average_influence'] for item in samples_to_plot]
    occurrences = [item['negative_occurrences'] for item in samples_to_plot]
    
    # 縮短樣本 ID 以便顯示
    shortened_ids = [f"{id[:15]}..." if len(id) > 18 else id for id in sample_ids]
    
    # 創建條形圖
    bars = plt.barh(range(len(shortened_ids)), avg_influences, alpha=0.8, color='crimson')
    
    # 在條形上標註出現次數
    for i, (bar, occ) in enumerate(zip(bars, occurrences)):
        plt.text(
            bar.get_width() - 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{occ}",
            ha='right',
            va='center',
            color='white',
            fontweight='bold'
        )
    
    # 設置 Y 軸標籤
    plt.yticks(range(len(shortened_ids)), shortened_ids)
    
    # 添加圖表標題和標籤
    plt.title(title)
    plt.xlabel('Average Negative Influence')
    plt.ylabel('Sample ID')
    plt.grid(True, axis='x', alpha=0.3)
    
    # 調整佈局
    plt.tight_layout()
    
    # 保存圖表
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Harmful samples plot saved to: {save_path}")
    
    plt.close()


def plot_sample_influence_heatmap(
    metadata_file: str,
    score_prefix: str = "tracin_influence_",
    save_path: Optional[str] = None,
    title: str = "Training Samples Influence on Test Samples",
    figsize: Tuple[int, int] = (14, 10),
    max_samples: int = 20
):
    """
    繪製訓練樣本對測試樣本的影響力熱圖。
    
    Args:
        metadata_file: 影響力元數據文件路徑
        score_prefix: 影響力分數的前綴
        save_path: 保存圖表的路徑（可選）
        title: 圖表標題
        figsize: 圖表尺寸
        max_samples: 要顯示的最大樣本數
    """
    # 加載元數據
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # 提取測試樣本 ID 和訓練樣本 ID
    test_sample_ids = set()
    for sample_id, sample_data in metadata.items():
        for key in sample_data.keys():
            if key.startswith(score_prefix):
                test_id = key[len(score_prefix):]
                test_sample_ids.add(test_id)
    
    # 限制樣本數量
    train_sample_ids = list(metadata.keys())[:max_samples]
    test_sample_ids = list(test_sample_ids)[:max_samples]
    
    # 創建影響力矩陣
    influence_matrix = np.zeros((len(train_sample_ids), len(test_sample_ids)))
    
    # 填充影響力矩陣
    for i, train_id in enumerate(train_sample_ids):
        for j, test_id in enumerate(test_sample_ids):
            influence_key = f"{score_prefix}{test_id}"
            if train_id in metadata and influence_key in metadata[train_id]:
                influence_matrix[i, j] = metadata[train_id][influence_key]
    
    # 創建熱圖
    plt.figure(figsize=figsize)
    sns.heatmap(
        influence_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        xticklabels=[f"{id[:10]}..." if len(id) > 13 else id for id in test_sample_ids],
        yticklabels=[f"{id[:10]}..." if len(id) > 13 else id for id in train_sample_ids],
        cbar_kws={'label': 'Influence Score'}
    )
    
    # 添加圖表標題和標籤
    plt.title(title)
    plt.xlabel('Test Samples')
    plt.ylabel('Training Samples')
    
    # 調整佈局
    plt.tight_layout()
    
    # 保存圖表
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Influence heatmap saved to: {save_path}")
    
    plt.close() 