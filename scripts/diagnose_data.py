#!/usr/bin/env python
"""
數據診斷腳本 (簡化版)

此腳本演示如何使用數據診斷模組來識別問題樣本。

用法:
    python scripts/diagnose_data.py --frequency <頻率> --material <材質> [options]
    
示例:
    python scripts/diagnose_data.py --frequency 1000hz --material plastic --detector gradient
    python scripts/diagnose_data.py --frequency 1000hz --material plastic --visualize
"""

import os
import sys
import argparse
import torch
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

# 添加項目根目錄到Python路徑以確保import正常工作
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from datasets import SpectrogramDatasetWithMaterial, RankingPairDataset
from models.resnet_ranker import SimpleCNNAudioRanker as ResNetAudioRanker
from utils.common_utils import set_seed

# 導入診斷模組
from utils.data_diagnostics import (
    ProblemSampleDetector,
    DiagnosticsVisualizer
)

def parse_arguments():
    """解析命令行參數。"""
    parser = argparse.ArgumentParser(description="數據診斷工具")
    
    # 數據選擇參數
    parser.add_argument('--frequency', type=str, required=True,
                        choices=['500hz', '1000hz', '3000hz'],
                        help='使用哪個頻率的數據進行診斷')
    
    parser.add_argument('--material', type=str, required=True,
                        choices=['box', 'plastic'],
                        help='使用哪種材質的數據進行診斷')
    
    # 診斷方法參數
    parser.add_argument('--detector', type=str, default='all',
                        choices=['gradient', 'feature', 'error', 'all'],
                        help='要使用的問題樣本檢測方法 (默認: all)')
    
    parser.add_argument('--gradient-threshold', type=float, default=0.95,
                        help='梯度異常檢測的閾值 (默認: 0.95)')
    
    parser.add_argument('--feature-method', type=str, default='isolation_forest',
                        choices=['isolation_forest', 'lof', 'dbscan'],
                        help='特徵空間離群檢測方法 (默認: isolation_forest)')
    
    # 可視化參數
    parser.add_argument('--visualize', action='store_true',
                        help='生成可視化結果')
    
    # 其他參數
    parser.add_argument('--model-path', type=str, default=None,
                        help='用於診斷的已訓練模型路徑，如果為None則將訓練新模型')
    
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批次大小 (默認: 32)')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='隨機種子 (默認: 42)')
    
    parser.add_argument('--output-dir', type=str, default=None,
                        help='輸出目錄，如果為None則自動創建')
    
    return parser.parse_args()

def create_output_dirs(args):
    """創建輸出目錄。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"diagnostics_results_{timestamp}"
    
    if args.output_dir is None:
        output_dir = os.path.join("scripts", base_name)
    else:
        output_dir = args.output_dir
        
    os.makedirs(output_dir, exist_ok=True)
    
    # 創建子目錄
    vis_dir = os.path.join(output_dir, "visualizations")
    log_dir = os.path.join(output_dir, "logs")
    
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    return {
        "base": output_dir,
        "vis": vis_dir,
        "logs": log_dir
    }

def load_or_train_model(args, dataset, device):
    """加載或訓練模型。"""
    # 如果提供了模型路徑，則加載模型
    if args.model_path is not None and os.path.exists(args.model_path):
        print(f"加載模型: {args.model_path}")
        model = ResNetAudioRanker(n_freqs=dataset.data.shape[2])
        model.load_state_dict(torch.load(args.model_path, map_location=device)['model_state_dict'])
        model.to(device)
        model.eval()
        return model
    
    # 否則訓練一個簡單模型用於診斷
    print("未提供模型路徑，訓練一個簡單模型用於診斷...")
    
    # 分割數據集
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    # 創建排名對數據集
    train_ranking_dataset = RankingPairDataset(train_dataset)
    val_ranking_dataset = RankingPairDataset(val_dataset)
    
    # 創建數據加載器
    train_loader = DataLoader(
        train_ranking_dataset,
        batch_size=min(args.batch_size, len(train_ranking_dataset)),
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_ranking_dataset,
        batch_size=min(args.batch_size, len(val_ranking_dataset)),
        shuffle=False,
        num_workers=2
    )
    
    # 初始化模型
    model = ResNetAudioRanker(n_freqs=dataset.data.shape[2])
    model.to(device)
    
    # 優化器和損失函數
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MarginRankingLoss(margin=1.0)
    
    # 訓練循環
    print("開始訓練簡單模型...")
    model.train()
    for epoch in range(5):
        for data1, data2, targets, _, _ in train_loader:
            data1, data2, targets = data1.to(device), data2.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            outputs1 = model(data1).view(-1)
            outputs2 = model(data2).view(-1)
            
            loss = criterion(outputs1, outputs2, targets)
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1}/5 完成")
    
    print("簡單模型訓練完成")
    model.eval()
    return model

def diagnose_problems(model, dataset, args, output_dirs):
    """診斷問題樣本。"""
    print("\n=== 開始問題樣本診斷 ===")
    
    # 創建問題樣本檢測器
    detector_config = {
        'output_dir': output_dirs['logs']
    }
    detector = ProblemSampleDetector(model, dataset, detector_config)
    
    # 根據選擇的檢測方法運行檢測
    if args.detector == 'gradient' or args.detector == 'all':
        print("\n檢測梯度異常樣本...")
        gradient_results = detector.detect_gradient_anomalies(threshold=args.gradient_threshold)
        print(f"發現 {len(gradient_results['anomaly_indices'])} 個梯度異常樣本")
        
        if len(gradient_results['anomaly_indices']) > 0:
            print(f"異常樣本索引: {gradient_results['anomaly_indices'][:10]}...")
            print(f"對應梯度值: {[float(f'{g:.4f}') for g in gradient_results['gradient_values'][:10]]}...")
    
    if args.detector == 'feature' or args.detector == 'all':
        print("\n檢測特徵空間離群樣本...")
        feature_results = detector.detect_feature_space_outliers(method=args.feature_method)
        print(f"發現 {len(feature_results['outlier_indices'])} 個特徵空間離群樣本")
        
        if len(feature_results['outlier_indices']) > 0:
            print(f"離群樣本索引: {feature_results['outlier_indices'][:10]}...")
            print(f"對應離群分數: {[float(f'{s:.4f}') for s in feature_results['outlier_scores'][:10]]}...")
        
    if args.detector == 'error' or args.detector == 'all':
        print("\n檢測一致性錯誤樣本...")
        error_results = detector.detect_consistent_errors(n_folds=3)
        print(f"發現 {len(error_results['error_indices'])} 個一致性錯誤樣本")
        
        if len(error_results['error_indices']) > 0:
            print(f"錯誤樣本索引: {error_results['error_indices'][:10]}...")
            print(f"對應錯誤率: {[float(f'{r:.4f}') for r in error_results['error_rates'][:10]]}...")
    
    if args.detector == 'all':
        # 運行綜合檢測
        print("\n運行綜合問題樣本檢測...")
        comprehensive_results = detector.run_comprehensive_detection()
        
        # 獲取問題樣本排名
        problem_ranking = detector.get_problem_samples_ranking()
        if len(problem_ranking) > 0:
            print(f"按問題嚴重程度排序的前10個樣本: {problem_ranking[:10]}")
    
    # 如果要求可視化，則創建可視化器並生成可視化
    if args.visualize:
        print("\n生成可視化結果...")
        visualizer_config = {
            'output_dir': output_dirs['vis']
        }
        visualizer = DiagnosticsVisualizer(detector, dataset, visualizer_config)
        
        try:
            # 生成特徵空間可視化
            print("\n生成特徵空間可視化...")
            feature_fig = visualizer.visualize_feature_space()
            feature_path = os.path.join(output_dirs['vis'], "feature_space_visualization.png")
            visualizer.save_visualization(feature_fig, feature_path)
            print(f"特徵空間可視化已保存至: {feature_path}")
        except Exception as e:
            print(f"無法生成特徵空間可視化: {e}")
        
        # 只在執行梯度檢測時可視化梯度分布
        if args.detector == 'gradient' or args.detector == 'all':
            try:
                print("\n生成梯度分布可視化...")
                gradient_fig = visualizer.visualize_gradient_distribution()
                gradient_path = os.path.join(output_dirs['vis'], "gradient_distribution.png")
                visualizer.save_visualization(gradient_fig, gradient_path)
                print(f"梯度分布可視化已保存至: {gradient_path}")
            except Exception as e:
                print(f"無法生成梯度分布可視化: {e}")
    
    # 保存檢測結果
    results_path = os.path.join(output_dirs['logs'], "detection_results.json")
    detector.save_results(results_path)
    print(f"檢測結果已保存至: {results_path}")
    
    return detector

def main():
    """主函數。"""
    # 解析命令行參數
    args = parse_arguments()
    
    # 設置隨機種子以確保可重現性
    set_seed(args.seed)
    print(f"使用隨機種子: {args.seed}")
    
    # 確定要使用的設備
    device = config.DEVICE
    print(f"使用設備: {device}")
    
    # 創建輸出目錄
    output_dirs = create_output_dirs(args)
    print(f"輸出目錄: {output_dirs['base']}")
    
    # 加載數據集
    print(f"\n加載數據集: {args.frequency}, 材質: {args.material}")
    try:
        dataset = SpectrogramDatasetWithMaterial(
            config.DATA_ROOT,
            config.CLASSES,
            config.SEQ_NUMS,
            args.frequency,
            args.material
        )
    except Exception as e:
        print(f"加載數據集時出錯: {e}")
        return
    
    if len(dataset) == 0:
        print("數據集為空，無法進行診斷。")
        return
    
    print(f"數據集大小: {len(dataset)} 樣本")
    
    # 加載或訓練模型
    model = load_or_train_model(args, dataset, device)
    
    # 診斷問題樣本
    detector = diagnose_problems(model, dataset, args, output_dirs)
    
    print(f"\n=== 診斷完成 ===\n結果保存在: {output_dirs['base']}")

if __name__ == "__main__":
    main() 