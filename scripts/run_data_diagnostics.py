#!/usr/bin/env python
"""
數據診斷腳本：用於分析頻譜圖數據集的品質和特性
功能：
- 加載數據集並進行診斷分析
- 生成診斷報告和可視化圖表
- 支持標準分類和排序數據集的分析
"""

import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 添加項目根目錄到 Python 路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 導入項目模塊
try:
    # 嘗試直接導入
    from datasets import SpectrogramDatasetWithMaterial, RankingPairDataset
except ImportError:
    # 如果失敗，嘗試從根目錄導入
    print("嘗試從根目錄導入 datasets 模塊...")
    import importlib.util
    import os
    
    # 構建 datasets.py 的絕對路徑
    datasets_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets.py')
    
    if os.path.exists(datasets_path):
        # 動態加載 datasets.py 模塊
        spec = importlib.util.spec_from_file_location("datasets_module", datasets_path)
        datasets_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(datasets_module)
        
        # 從加載的模塊中獲取所需的類
        SpectrogramDatasetWithMaterial = datasets_module.SpectrogramDatasetWithMaterial
        RankingPairDataset = datasets_module.RankingPairDataset
        print("成功從根目錄加載 datasets 模塊。")
    else:
        print(f"錯誤: 找不到 datasets.py 文件。路徑: {datasets_path}")
        sys.exit(1)

from config import CLASSES, FREQUENCIES, MATERIAL, DATA_ROOT, SEQ_NUMS
from utils.data_diagnostics import (
    analyze_spectrogram_statistics,
    plot_spectrogram_distribution,
    detect_spectrogram_anomalies,
    analyze_class_distribution,
    analyze_feature_importance,
    cross_validate_feature_robustness,
    examine_data_leakage,
    analyze_ranking_pairs,
    evaluate_ranking_consistency,
    visualize_ranking_embeddings
)

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description="音頻頻譜圖數據診斷工具")
    
    parser.add_argument("--data-dir", type=str, default=DATA_ROOT, 
                        help="數據集根目錄路徑")
    parser.add_argument("--material", type=str, default=MATERIAL,
                        help="要使用的材質類型 (如 'box', 'plastic')")
    parser.add_argument("--frequency", type=str, default=FREQUENCIES[0],
                        help="要使用的頻率 (如 '1000hz')")
    parser.add_argument("--output-dir", type=str, default="diagnostics_results",
                        help="診斷結果的輸出目錄")
    parser.add_argument("--ranking", action="store_true",
                        help="是否分析排序數據集 (而非標準分類數據集)")
    parser.add_argument("--anomaly-threshold", type=float, default=0.05,
                        help="異常樣本檢測的閾值 (0-0.5)")
    
    args = parser.parse_args()
    return args

def run_diagnostics():
    """執行數據診斷"""
    args = parse_args()
    
    # 配置診斷結果的輸出目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 配置子目錄
    spectrogram_dir = os.path.join(output_dir, "spectrogram_analysis")
    dataset_dir = os.path.join(output_dir, "dataset_analysis")
    ranking_dir = os.path.join(output_dir, "ranking_analysis")
    
    os.makedirs(spectrogram_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    if args.ranking:
        os.makedirs(ranking_dir, exist_ok=True)
    
    # 加載數據集
    print(f"正在加載數據集 (材質: {args.material}, 頻率: {args.frequency})...")
    dataset = SpectrogramDatasetWithMaterial(
        data_dir=args.data_dir,
        classes=CLASSES,
        selected_seqs=SEQ_NUMS,
        selected_freq=args.frequency,
        material=args.material
    )
    
    if len(dataset) == 0:
        print("錯誤: 數據集為空。請檢查路徑和參數設置。")
        return
    
    print(f"成功加載數據集，共 {len(dataset)} 個樣本。")
    
    # 提取數據和標籤
    if args.ranking:
        print("\n正在創建排序對數據集...")
        ranking_dataset = RankingPairDataset(dataset)
        
        # 獲取排序對樣本
        # 由於排序對數據集返回的是 (data1, data2, target, label1, label2)
        # 因此需要一個單獨的循環來提取數據
        data1_list, data2_list, targets_list, labels1_list, labels2_list = [], [], [], [], []
        
        # 只提取一部分樣本進行分析 (如果排序對太多)
        max_pairs = min(1000, len(ranking_dataset))
        indices = np.random.choice(len(ranking_dataset), max_pairs, replace=False)
        
        for idx in indices:
            data1, data2, target, label1, label2 = ranking_dataset[idx]
            data1_list.append(data1.unsqueeze(0))  # 添加批次維度
            data2_list.append(data2.unsqueeze(0))
            targets_list.append(target)
            labels1_list.append(label1)
            labels2_list.append(label2)
        
        data1_batch = torch.cat(data1_list, dim=0)
        data2_batch = torch.cat(data2_list, dim=0)
        targets = torch.tensor(targets_list)
        labels1 = torch.tensor(labels1_list)
        labels2 = torch.tensor(labels2_list)
        
        print(f"已創建 {max_pairs} 個排序對用於診斷分析。")
        
        # 1. 分析頻譜圖統計特性 (使用第一組數據)
        print("\n===== 頻譜圖統計分析 =====")
        analyze_spectrogram_statistics(data1_batch, save_dir=spectrogram_dir)
        
        # 2. 繪製頻譜圖分佈
        print("\n===== 頻譜圖分佈分析 =====")
        plot_spectrogram_distribution(data1_batch, labels1, CLASSES, save_dir=spectrogram_dir)
        
        # 3. 檢測異常頻譜圖
        print("\n===== 異常頻譜圖檢測 =====")
        detect_spectrogram_anomalies(data1_batch, contamination=args.anomaly_threshold, save_dir=spectrogram_dir)
        
        # 4. 分析類別分佈
        print("\n===== 類別分佈分析 =====")
        analyze_class_distribution(labels1, CLASSES, save_dir=dataset_dir)
        
        # 5. 分析排序對
        print("\n===== 排序對分析 =====")
        analyze_ranking_pairs(data1_batch, data2_batch, targets, labels1, labels2, save_dir=ranking_dir)
        
        # 6. 評估排序一致性
        print("\n===== 排序一致性分析 =====")
        evaluate_ranking_consistency(targets, labels1, labels2, save_dir=ranking_dir)
        
        # 7. 可視化排序嵌入
        print("\n===== 排序嵌入可視化 =====")
        visualize_ranking_embeddings(data1_batch, labels1, save_dir=ranking_dir)
        
    else:
        # 標準分類數據集分析
        data = dataset.data
        labels = dataset.labels
        
        # 1. 分析頻譜圖統計特性
        print("\n===== 頻譜圖統計分析 =====")
        analyze_spectrogram_statistics(data, save_dir=spectrogram_dir)
        
        # 2. 繪製頻譜圖分佈
        print("\n===== 頻譜圖分佈分析 =====")
        plot_spectrogram_distribution(data, labels, CLASSES, save_dir=spectrogram_dir)
        
        # 3. 檢測異常頻譜圖
        print("\n===== 異常頻譜圖檢測 =====")
        detect_spectrogram_anomalies(data, contamination=args.anomaly_threshold, save_dir=spectrogram_dir)
        
        # 4. 分析類別分佈
        print("\n===== 類別分佈分析 =====")
        analyze_class_distribution(labels, CLASSES, save_dir=dataset_dir)
        
        # 5. 分析特徵重要性
        print("\n===== 特徵重要性分析 =====")
        analyze_feature_importance(data, labels, save_dir=dataset_dir)
        
        # 6. 交叉驗證特徵穩健性
        print("\n===== 特徵穩健性交叉驗證 =====")
        cross_validate_feature_robustness(data, labels, save_dir=dataset_dir)
        
        # 7. 檢查數據洩漏
        print("\n===== 數據洩漏檢查 =====")
        # 隨機分割訓練和測試集
        n_samples = len(data)
        indices = np.random.permutation(n_samples)
        split = int(n_samples * 0.8)
        train_indices = indices[:split]
        test_indices = indices[split:]
        
        examine_data_leakage(data, labels, train_indices, test_indices, save_dir=dataset_dir)
    
    print(f"\n數據診斷分析完成！結果已保存到 {output_dir}")

if __name__ == "__main__":
    run_diagnostics() 