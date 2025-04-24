#!/usr/bin/env python
"""
識別持續困難的音檔樣本
這個腳本分析元數據文件，找出那些在多個epoch中持續被分類為困難樣本(高bin值)的音檔
"""

import json
import os
import glob
from collections import defaultdict

# 搜索最新的元數據文件
def find_latest_metadata_file():
    """找到最新的元數據文件"""
    # 首先嘗試在saved_models/metadata目錄下找
    metadata_pattern = "saved_models/metadata/metadata_v*.json"
    metadata_files = glob.glob(metadata_pattern)
    
    # 如果沒找到，嘗試原始位置
    if not metadata_files:
        metadata_pattern = "/Users/sbplab/Hank/sinetone_sliced/step_018_sliced/metadata/metadata_v*.json"
        metadata_files = glob.glob(metadata_pattern)
    
    if not metadata_files:
        return "/Users/sbplab/Hank/sinetone_sliced/step_018_sliced/metadata/metadata_v1.0.json"
    
    # 按修改時間排序
    latest_file = max(metadata_files, key=os.path.getmtime)
    return latest_file

# 元數據文件路徑
METADATA_FILE = find_latest_metadata_file()
# 定義困難樣本的bin索引，對於10個bin，最困難的是bin 9
DIFFICULT_BIN = 9
# 定義樣本被視為"持續困難"所需的最小epoch數量 (針對30個epoch調整)
MIN_EPOCHS_AS_DIFFICULT = 6  # 約20%的訓練周期
# 定義高度困難的閾值 (針對30個epoch調整)
HIGH_DIFFICULTY_THRESHOLD = 12  # 約40%的訓練周期

def find_persistent_difficult_samples():
    """識別持續困難的音檔樣本"""
    if not os.path.exists(METADATA_FILE):
        print(f"元數據文件不存在: {METADATA_FILE}")
        return

    # 加載元數據
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    
    print(f"使用元數據文件: {METADATA_FILE}")
    print(f"元數據中的樣本總數: {len(metadata)}")
    
    # 追踪每個樣本在困難bin中出現的次數
    difficult_sample_counts = defaultdict(int)
    # 追踪每個樣本的bin分配歷史
    sample_bin_history = {}
    # 記錄所有epoch
    all_epochs = set()
    
    for sample_id, info in metadata.items():
        ghm_bins = info.get('ghm_bins', {})
        if ghm_bins:
            # 初始化bin歷史記錄
            sample_bin_history[sample_id] = {}
            
            for epoch, bin_idx in ghm_bins.items():
                all_epochs.add(int(epoch))
                sample_bin_history[sample_id][epoch] = bin_idx
                
                # 檢查是否為困難樣本
                if bin_idx == DIFFICULT_BIN:
                    difficult_sample_counts[sample_id] += 1
    
    # 獲取最大epoch
    max_epoch = max(all_epochs) if all_epochs else 0
    print(f"找到訓練數據總共包含 {max_epoch} 個epoch")
    
    # 過濾出持續困難的樣本
    persistent_difficult_samples = [
        sample_id for sample_id, count in difficult_sample_counts.items()
        if count >= MIN_EPOCHS_AS_DIFFICULT
    ]
    
    # 過濾出高度困難的樣本
    high_difficulty_samples = [
        sample_id for sample_id, count in difficult_sample_counts.items()
        if count >= HIGH_DIFFICULTY_THRESHOLD
    ]
    
    # 按困難頻率排序
    persistent_difficult_samples.sort(
        key=lambda x: difficult_sample_counts[x], 
        reverse=True
    )
    
    print(f"\n持續困難的樣本數量: {len(persistent_difficult_samples)} (出現在bin {DIFFICULT_BIN} 至少 {MIN_EPOCHS_AS_DIFFICULT} 次)")
    print(f"高度困難的樣本數量: {len(high_difficulty_samples)} (出現在bin {DIFFICULT_BIN} 至少 {HIGH_DIFFICULTY_THRESHOLD} 次)")
    
    print("\n持續困難的樣本列表:")
    print("=" * 100)
    print(f"{'樣本ID':<40} {'困難epoch數':<15} {'佔比':<10} {'所有bin歷史'}")
    print("-" * 100)
    
    # 角度統計
    angle_stats = defaultdict(int)
    
    for sample_id in persistent_difficult_samples:
        count = difficult_sample_counts[sample_id]
        percentage = (count / max_epoch) * 100
        
        # 構建bin歷史字符串
        bin_history = []
        for epoch in range(1, max_epoch + 1):
            epoch_str = str(epoch)
            if epoch_str in sample_bin_history[sample_id]:
                bin_history.append(str(sample_bin_history[sample_id][epoch_str]))
            else:
                bin_history.append('-')
        
        bin_history_str = '[' + ', '.join(bin_history) + ']'
        
        # 提取樣本的詳細信息
        sample_info = metadata[sample_id]
        angle = sample_info.get('angle', 'N/A')
        material = sample_info.get('material', 'N/A')
        
        # 更新角度統計
        angle_stats[angle] += 1
        
        # 高度困難樣本標記
        difficulty_marker = " [高度困難]" if sample_id in high_difficulty_samples else ""
        
        print(f"{sample_id:<40} {count:<15} {percentage:>6.1f}%  {bin_history_str}{difficulty_marker}")
        print(f"  角度: {angle}, 材料: {material}")
    
    print("=" * 100)
    print(f"注: bin歷史格式為 [epoch1的bin, epoch2的bin, ..., epochN的bin]")
    print(f"    '-' 表示該epoch中沒有記錄")
    print(f"    bin值越高表示樣本越困難，bin {DIFFICULT_BIN} 為最困難樣本")
    
    # 打印角度分布統計
    total_difficult = len(persistent_difficult_samples)
    if total_difficult > 0:
        print("\n困難樣本角度分布:")
        print("-" * 50)
        for angle in sorted(angle_stats.keys()):
            count = angle_stats[angle]
            percentage = (count / total_difficult) * 100
            print(f"角度 {angle}°: {count} 個樣本 ({percentage:.1f}%)")

if __name__ == "__main__":
    find_persistent_difficult_samples() 