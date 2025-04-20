#!/usr/bin/env python
"""
分析每個檢查點的TracIn影響分數，追蹤訓練過程中影響分數的變化。

此腳本可以識別：
1. 每個檢查點中對測試樣本有最大影響的訓練樣本
2. 訓練過程中影響分數變化最顯著的訓練樣本
3. 不同檢查點之間影響分數的趨勢變化
"""

import os
import json
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def extract_degrees(sample_id):
    """從樣本ID中提取角度信息。"""
    parts = sample_id.split("_")
    first_deg, second_deg = 0, 0
    
    # 尋找包含"deg"的部分，並提取角度值
    for part in parts:
        if part.startswith("deg"):
            try:
                deg_value = int(part[3:])  # 提取"deg"後的數字
                if first_deg == 0:
                    first_deg = deg_value
                else:
                    second_deg = deg_value
                    break
            except ValueError:
                continue
    
    if first_deg == 0 and second_deg == 0:
        print(f"警告: 無法從樣本ID '{sample_id}'提取角度值")
    
    return first_deg, second_deg


def extract_checkpoint_epoch(checkpoint_name):
    """從檢查點名稱中提取epoch編號。"""
    match = re.search(r'epoch_(\d+)\.pt', checkpoint_name)
    if match:
        return int(match.group(1))
    else:
        # 嘗試其他可能的格式
        match = re.search(r'(\d+)\.pt', checkpoint_name)
        if match:
            return int(match.group(1))
        return -1  # 無法提取epoch編號


def parse_args():
    """解析命令行參數。"""
    parser = argparse.ArgumentParser(description="分析每個檢查點的TracIn影響分數")
    parser.add_argument("--metadata-dir", type=str, default="/Users/sbplab/Hank/sinetone_sliced/step_018_sliced/metadata",
                        help="元數據目錄")
    parser.add_argument("--material", type=str, default="plastic",
                        help="材料類型")
    parser.add_argument("--frequency", type=str, default="500hz",
                        help="頻率數據")
    parser.add_argument("--test-sample", type=str, default=None,
                        help="要分析的特定測試樣本ID (如果不指定，將選擇前幾個)")
    parser.add_argument("--num-test-samples", type=int, default=3,
                        help="要分析的測試樣本數量 (如果不指定特定樣本)")
    parser.add_argument("--top-n", type=int, default=5,
                        help="每個檢查點顯示的前N個影響樣本")
    parser.add_argument("--save-plots", action="store_true",
                        help="將圖表保存為PNG文件而不是顯示")
    parser.add_argument("--output-dir", type=str, default="tracin_plots",
                        help="保存圖表的目錄")
    return parser.parse_args()


def load_per_checkpoint_metadata(metadata_dir, material, frequency):
    """加載每個檢查點的影響分數元數據。"""
    metadata_file = os.path.join(
        metadata_dir, 
        f"{material}_{frequency}_per_checkpoint_influence_metadata.json"
    )
    
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"找不到每個檢查點的影響分數文件: {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        return json.load(f)


def identify_test_samples(metadata):
    """識別元數據中的測試樣本。"""
    test_samples = set()
    pattern = re.compile(r"tracin_influence_(.*?)_model_")
    
    for train_id, scores in metadata.items():
        for score_name in scores.keys():
            match = pattern.search(score_name)
            if match:
                test_samples.add(match.group(1))
    
    return sorted(list(test_samples))


def identify_checkpoints(metadata):
    """識別元數據中的檢查點。"""
    checkpoints = set()
    pattern = re.compile(r"_(model_.*?\.pt)$")
    
    for train_id, scores in metadata.items():
        for score_name in scores.keys():
            match = pattern.search(score_name)
            if match:
                checkpoints.add(match.group(1))
    
    # 按照epoch編號排序檢查點
    return sorted(list(checkpoints), key=extract_checkpoint_epoch)


def find_influential_samples_per_checkpoint(metadata, test_sample, checkpoints, top_n=5):
    """為測試樣本找出每個檢查點中最具影響力的訓練樣本。"""
    results = {}
    
    for checkpoint in checkpoints:
        # 收集此檢查點對測試樣本的所有影響分數
        influences = []
        score_prefix = f"tracin_influence_{test_sample}_{checkpoint}"
        
        for train_id, scores in metadata.items():
            for score_name, score in scores.items():
                if score_name == score_prefix:
                    influences.append((train_id, score))
        
        # 按絕對影響力排序
        influences_by_abs = sorted(influences, key=lambda x: abs(x[1]), reverse=True)
        
        # 按正面影響力排序
        influences_positive = sorted(influences, key=lambda x: x[1], reverse=True)
        
        # 按負面影響力排序
        influences_negative = sorted(influences, key=lambda x: x[1])
        
        results[checkpoint] = {
            "top_by_abs": influences_by_abs[:top_n],
            "top_positive": influences_positive[:top_n],
            "top_negative": influences_negative[:top_n]
        }
    
    return results


def find_samples_with_changing_influence(results, checkpoints):
    """找出在訓練過程中影響力變化最顯著的樣本。"""
    # 追蹤每個訓練樣本在所有檢查點中的影響力變化
    sample_influences = defaultdict(list)
    
    for checkpoint in checkpoints:
        # 記錄所有出現在top_by_abs中的樣本
        for train_id, influence in results[checkpoint]["top_by_abs"]:
            sample_influences[train_id].append((checkpoint, influence))
    
    # 計算每個樣本的影響力變化範圍
    sample_changes = {}
    for train_id, influences in sample_influences.items():
        if len(influences) > 1:  # 至少在兩個檢查點中出現
            values = [infl for _, infl in influences]
            sample_changes[train_id] = max(values) - min(values)
    
    # 按變化幅度排序
    sorted_changes = sorted(sample_changes.items(), key=lambda x: abs(x[1]), reverse=True)
    
    return sorted_changes[:10], sample_influences


def plot_influence_trends(test_sample, sample_influences, checkpoints, args):
    """繪製訓練樣本對測試樣本的影響力變化趨勢。"""
    plt.figure(figsize=(12, 8))
    
    # 選擇影響力變化最大的前N個樣本
    top_changing_samples = sorted(
        sample_influences.items(), 
        key=lambda x: max([abs(infl) for _, infl in x[1]]),
        reverse=True
    )[:8]
    
    # 簡化檢查點名稱 (僅保留epoch編號)
    checkpoint_epochs = [extract_checkpoint_epoch(cp) for cp in checkpoints]
    
    # 為每個樣本構建完整的影響力序列 (在所有檢查點上)
    for train_id, influences in top_changing_samples:
        # 創建一個影響力字典，用檢查點作為鍵
        influence_dict = {checkpoint: 0 for checkpoint in checkpoints}
        for checkpoint, infl in influences:
            influence_dict[checkpoint] = infl
        
        # 提取角度
        train_deg1, train_deg2 = extract_degrees(train_id)
        
        # 繪製影響力變化趨勢
        plt.plot(
            checkpoint_epochs,
            [influence_dict[cp] for cp in checkpoints],
            marker='o',
            label=f"{train_id} [{train_deg1}→{train_deg2}]"
        )
    
    test_deg1, test_deg2 = extract_degrees(test_sample)
    plt.title(f"測試樣本 {test_sample} [{test_deg1}→{test_deg2}] 的影響力變化趨勢")
    plt.xlabel("訓練階段 (Epoch)")
    plt.ylabel("影響力分數")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if args.save_plots:
        # 創建輸出目錄
        os.makedirs(args.output_dir, exist_ok=True)
        # 保存圖表
        plt.savefig(os.path.join(args.output_dir, f"influence_trends_{test_sample}.png"), dpi=300)
    else:
        plt.show()


def print_results_per_checkpoint(test_sample, results, checkpoints):
    """打印每個檢查點的影響力分析結果。"""
    test_deg1, test_deg2 = extract_degrees(test_sample)
    print(f"\n==== 測試樣本: {test_sample} [角度: {test_deg1}→{test_deg2}] ====")
    
    for checkpoint in checkpoints:
        epoch = extract_checkpoint_epoch(checkpoint)
        print(f"\n  檢查點: {checkpoint} (Epoch {epoch})")
        
        print("    最大絕對影響:")
        for train_id, influence in results[checkpoint]["top_by_abs"]:
            train_deg1, train_deg2 = extract_degrees(train_id)
            print(f"      {train_id} [角度: {train_deg1}→{train_deg2}]: {influence:.4f}")
        
        print("    最大正面影響:")
        for train_id, influence in results[checkpoint]["top_positive"][:3]:
            if influence <= 0:
                continue
            train_deg1, train_deg2 = extract_degrees(train_id)
            print(f"      {train_id} [角度: {train_deg1}→{train_deg2}]: {influence:.4f}")
        
        print("    最大負面影響:")
        for train_id, influence in results[checkpoint]["top_negative"][:3]:
            if influence >= 0:
                continue
            train_deg1, train_deg2 = extract_degrees(train_id)
            print(f"      {train_id} [角度: {train_deg1}→{train_deg2}]: {influence:.4f}")


def print_changing_influences(changing_samples, sample_influences, checkpoints):
    """打印影響力變化最顯著的樣本。"""
    print("\n==== 影響力變化最顯著的訓練樣本 ====")
    
    for train_id, change in changing_samples:
        train_deg1, train_deg2 = extract_degrees(train_id)
        influences = sample_influences[train_id]
        
        # 整理影響力變化數據
        influence_dict = {}
        for checkpoint, infl in influences:
            epoch = extract_checkpoint_epoch(checkpoint)
            influence_dict[epoch] = infl
        
        # 按epoch排序
        sorted_epochs = sorted(influence_dict.keys())
        
        print(f"\n訓練樣本: {train_id} [角度: {train_deg1}→{train_deg2}]")
        print(f"  影響力變化幅度: {change:.4f}")
        print("  各檢查點影響力:")
        for epoch in sorted_epochs:
            print(f"    Epoch {epoch}: {influence_dict[epoch]:.4f}")


def main():
    """主函數。"""
    args = parse_args()
    
    # 加載每個檢查點的影響分數元數據
    metadata = load_per_checkpoint_metadata(args.metadata_dir, args.material, args.frequency)
    
    # 識別測試樣本
    all_test_samples = identify_test_samples(metadata)
    print(f"發現 {len(all_test_samples)} 個測試樣本")
    
    # 識別檢查點
    checkpoints = identify_checkpoints(metadata)
    print(f"發現 {len(checkpoints)} 個檢查點")
    
    # 選擇要分析的測試樣本
    test_samples_to_analyze = []
    if args.test_sample:
        if args.test_sample in all_test_samples:
            test_samples_to_analyze.append(args.test_sample)
        else:
            print(f"警告: 指定的測試樣本 '{args.test_sample}' 不在元數據中")
            test_samples_to_analyze = all_test_samples[:args.num_test_samples]
    else:
        test_samples_to_analyze = all_test_samples[:args.num_test_samples]
    
    # 分析每個測試樣本
    for test_sample in test_samples_to_analyze:
        # 找出每個檢查點中最具影響力的訓練樣本
        results = find_influential_samples_per_checkpoint(
            metadata, test_sample, checkpoints, top_n=args.top_n
        )
        
        # 找出影響力變化最顯著的樣本
        changing_samples, sample_influences = find_samples_with_changing_influence(
            results, checkpoints
        )
        
        # 打印每個檢查點的結果
        print_results_per_checkpoint(test_sample, results, checkpoints)
        
        # 打印影響力變化最顯著的樣本
        print_changing_influences(changing_samples, sample_influences, checkpoints)
        
        # 繪製影響力變化趨勢
        plot_influence_trends(test_sample, sample_influences, checkpoints, args)


if __name__ == "__main__":
    main() 