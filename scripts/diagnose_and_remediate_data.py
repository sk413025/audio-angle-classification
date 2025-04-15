#!/usr/bin/env python
"""
數據診斷與修正腳本

此腳本演示如何使用數據診斷和修正模組來識別和改善問題樣本。

用法:
    python scripts/diagnose_and_remediate_data.py --frequency <頻率> --material <材質> [options]
    
示例:
    python scripts/diagnose_and_remediate_data.py --frequency 1000hz --material plastic --detector gradient
    python scripts/diagnose_and_remediate_data.py --frequency 1000hz --remediate weighted --apply-to-train
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

# 導入診斷和修正模組
from utils.data_diagnostics import (
    ProblemSampleDetector,
    DiagnosticsVisualizer,
    RemediationStrategies
)

def parse_arguments():
    """解析命令行參數。"""
    parser = argparse.ArgumentParser(description="數據診斷與修正工具")
    
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
    
    # 修正方法參數
    parser.add_argument('--remediate', type=str, default=None,
                        choices=['relabel', 'weighted', 'augment', 'synthetic', 'remove'],
                        help='要應用的修正策略')
    
    parser.add_argument('--apply-to-train', action='store_true',
                        help='將修正策略應用到訓練模型上')
    
    parser.add_argument('--epochs', type=int, default=5,
                        help='如果應用到訓練上，要訓練的輪數 (默認: 5)')
    
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
    model_dir = os.path.join(output_dir, "models")
    
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    return {
        "base": output_dir,
        "vis": vis_dir,
        "logs": log_dir,
        "models": model_dir
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
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_ranking_dataset,
        batch_size=min(args.batch_size, len(val_ranking_dataset)),
        shuffle=False,
        num_workers=0
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
    
    if args.detector == 'feature' or args.detector == 'all':
        print("\n檢測特徵空間離群樣本...")
        feature_results = detector.detect_feature_space_outliers(method=args.feature_method)
        print(f"發現 {len(feature_results['outlier_indices'])} 個特徵空間離群樣本")
        
    if args.detector == 'error' or args.detector == 'all':
        print("\n檢測一致性錯誤樣本...")
        error_results = detector.detect_consistent_errors(n_folds=3)
        print(f"發現 {len(error_results['error_indices'])} 個一致性錯誤樣本")
    
    if args.detector == 'all':
        # 運行綜合檢測
        print("\n運行綜合問題樣本檢測...")
        comprehensive_results = detector.run_comprehensive_detection()
        # 獲取問題樣本排名
        problem_ranking = detector.get_problem_samples_ranking()
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
            feature_fig = visualizer.visualize_feature_space()
            visualizer.save_visualization(feature_fig, "feature_space_visualization.png")
            print("特徵空間可視化已保存")
        except Exception as e:
            print(f"無法生成特徵空間可視化: {e}")
        
        # 只在執行梯度檢測時可視化梯度分布
        if args.detector == 'gradient' or args.detector == 'all':
            try:
                gradient_fig = visualizer.visualize_gradient_distribution()
                visualizer.save_visualization(gradient_fig, "gradient_distribution.png")
                print("梯度分布可視化已保存")
            except Exception as e:
                print(f"無法生成梯度分布可視化: {e}")
        
        # 只在執行一致性錯誤檢測時可視化錯誤模式
        if (args.detector == 'error' or args.detector == 'all') and 'consistent_errors' in detector.results:
            try:
                error_fig = visualizer.visualize_error_patterns()
                visualizer.save_visualization(error_fig, "error_patterns.png")
                print("錯誤模式可視化已保存")
            except Exception as e:
                print(f"無法生成錯誤模式可視化: {e}")
                print("這可能是由於模型、數據格式或維度問題造成的")
        
        # 生成綜合報告
        if args.detector == 'all':
            try:
                report_path = visualizer.generate_comprehensive_report()
                print(f"綜合報告已保存至: {report_path}")
            except Exception as e:
                print(f"無法生成綜合報告: {e}")
    
    # 保存檢測結果
    results_path = os.path.join(output_dirs['logs'], "detection_results.json")
    detector.save_results(results_path)
    print(f"檢測結果已保存至: {results_path}")
    
    return detector

def apply_remediation(detector, dataset, args, output_dirs, device):
    """應用數據修正策略。"""
    if args.remediate is None:
        print("\n未指定修正策略，跳過修正階段")
        return None
    
    print(f"\n=== 應用 {args.remediate} 修正策略 ===")
    
    # 創建修正策略器
    remediation_config = {
        'log_dir': output_dirs['logs']
    }
    remediation = RemediationStrategies(detector, dataset, remediation_config)
    
    # 根據選擇的策略應用修正
    remediated_dataset = None
    
    if args.remediate == 'relabel':
        print("\n建議需要重新標記的樣本...")
        suggestions = remediation.suggest_relabeling(confidence_threshold=0.7)
        print(f"建議重新標記 {len(suggestions['samples'])} 個樣本")
        
        # 這裡只是示例，實際上不會自動重新標記
        print("在實際應用中，這些樣本需要由人工重新檢查和標記")
        
    elif args.remediate == 'weighted':
        print("\n生成基於樣本質量的權重...")
        weights = remediation.generate_sample_weights(method='inverse_difficulty')
        
        print("\n應用加權抽樣策略...")
        remediated_dataset = remediation.apply_remediation(
            dataset, strategy="weighted_sampling", weights=weights
        )
        
    elif args.remediate == 'augment':
        print("\n建議數據增強策略...")
        augmentation_strategies = remediation.suggest_augmentation_strategies()
        
        print("\n應用數據增強策略...")
        remediated_dataset = remediation.apply_remediation(
            dataset, strategy="augmentation", augmentation_strategies=augmentation_strategies
        )
        
    elif args.remediate == 'synthetic':
        print("\n生成合成樣本...")
        synthetic_data, synthetic_labels = remediation.generate_synthetic_samples()
        
        print("\n應用合成樣本策略...")
        remediated_dataset = remediation.apply_remediation(
            dataset, strategy="synthetic_samples", 
            synthetic_data=synthetic_data, synthetic_labels=synthetic_labels
        )
        
    elif args.remediate == 'remove':
        print("\n應用樣本移除策略...")
        remediated_dataset = remediation.apply_remediation(
            dataset, strategy="remove_samples"
        )
    
    # 如果指定了應用到訓練，則訓練模型並評估效果
    if args.apply_to_train and remediated_dataset is not None:
        print("\n=== 使用修正後的數據集訓練模型並評估效果 ===")
        
        # 創建新模型
        model = ResNetAudioRanker(n_freqs=dataset.data.shape[2])
        model.to(device)
        
        # 優化器和損失函數
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MarginRankingLoss(margin=1.0)
        
        # 如果是加權採樣策略，需要使用特殊的數據加載器
        if args.remediate == 'weighted':
            # 獲取加權採樣器
            from torch.utils.data import WeightedRandomSampler
            
            # 創建排名對數據集
            ranking_dataset = RankingPairDataset(remediated_dataset)
            
            # 從修正策略中獲取權重
            sample_weights = remediation.generate_sample_weights()
            
            # 轉換權重字典為權重列表
            if isinstance(sample_weights, dict):
                weight_list = [sample_weights.get(i, 1.0) for i in range(len(remediated_dataset))]
            else:
                weight_list = sample_weights
                
            sampler = WeightedRandomSampler(
                weights=weight_list,
                num_samples=len(ranking_dataset),
                replacement=True
            )
            
            train_loader = DataLoader(
                ranking_dataset,
                batch_size=args.batch_size,
                sampler=sampler,
                num_workers=0
            )
        else:
            # 其他策略使用普通數據加載器
            # 創建排名對數據集
            ranking_dataset = RankingPairDataset(remediated_dataset)
            
            train_loader = DataLoader(
                ranking_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0
            )
        
        # 訓練模型
        model.train()
        for epoch in range(args.epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            for data1, data2, targets, _, _ in train_loader:
                data1, data2, targets = data1.to(device), data2.to(device), targets.to(device)
                
                optimizer.zero_grad()
                
                outputs1 = model(data1).view(-1)
                outputs2 = model(data2).view(-1)
                
                loss = criterion(outputs1, outputs2, targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # 計算準確率（正確預測的排序對）
                predictions = (outputs1 > outputs2).float() * 2 - 1
                correct += (predictions == targets).sum().item()
                total += targets.size(0)
            
            avg_loss = total_loss / len(train_loader)
            accuracy = correct / total
            
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        print(f"\n修正後模型訓練完成")
        
        # 保存修正後的模型
        model_save_path = os.path.join(output_dirs['models'], f"remediated_model_{args.remediate}.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'remediation_strategy': args.remediate,
        }, model_save_path)
        print(f"修正後的模型已保存至: {model_save_path}")
    
    return remediated_dataset

def main():
    """主函數。"""
    # 解析命令行參數
    args = parse_arguments()
    
    # 設置隨機種子
    set_seed(args.seed)
    
    # 創建輸出目錄
    output_dirs = create_output_dirs(args)
    print(f"輸出目錄: {output_dirs['base']}")
    
    # 設置設備（GPU或CPU）
    device = config.DEVICE
    print(f"使用設備: {device}")
    
    # 加載數據集
    print(f"\n加載 {args.frequency} 頻率和 {args.material} 材質的數據...")
    dataset = SpectrogramDatasetWithMaterial(
        data_dir=config.DATA_ROOT,
        classes=config.CLASSES,
        selected_seqs=config.SEQ_NUMS,
        selected_freq=args.frequency,
        material=args.material
    )
    print(f"加載了 {len(dataset)} 個樣本")
    
    # 加載或訓練模型
    model = load_or_train_model(args, dataset, device)
    
    # 診斷問題樣本
    detector = diagnose_problems(model, dataset, args, output_dirs)
    
    # 如果指定了修正策略，則應用修正
    remediated_dataset = apply_remediation(detector, dataset, args, output_dirs, device)
    
    print("\n=== 數據診斷與修正完成 ===")
    print(f"結果已保存到: {output_dirs['base']}")

if __name__ == "__main__":
    main()