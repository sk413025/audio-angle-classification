#!/usr/bin/env python
"""
生成困難樣本的排除列表
"""

import json
import os
from collections import defaultdict
import datetime

# 元數據文件路徑
METADATA_FILE = "/Users/sbplab/Hank/sinetone_sliced/step_018_sliced/metadata/metadata_v1.0.json"
# 排除列表輸出路徑
OUTPUT_DIR = "exclusion_lists"
# 定義困難樣本的bin索引
DIFFICULT_BIN = 0
# 定義樣本被視為"持續困難"所需的最小epoch數量
MIN_EPOCHS_AS_DIFFICULT = 2

def generate_exclusion_lists():
    """生成困難樣本的排除列表"""
    
    if not os.path.exists(METADATA_FILE):
        print(f"元數據文件不存在: {METADATA_FILE}")
        return
    
    # 創建輸出目錄
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加載元數據
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    
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
    
    # 生成不同的排除列表
    
    # 1. 所有持續困難樣本
    persistent_difficult_samples = [
        sample_id for sample_id, count in difficult_sample_counts.items()
        if count >= MIN_EPOCHS_AS_DIFFICULT
    ]
    
    # 2. 高度困難樣本 (在超過50%的epoch中為困難樣本)
    high_difficulty_threshold = max_epoch * 0.5
    high_difficulty_samples = [
        sample_id for sample_id, count in difficult_sample_counts.items()
        if count >= high_difficulty_threshold
    ]
    
    # 3. 按角度分類的困難樣本
    samples_by_angle = defaultdict(list)
    for sample_id in persistent_difficult_samples:
        angle = metadata[sample_id].get('angle', 'unknown')
        samples_by_angle[angle].append(sample_id)
    
    # 寫入排除列表文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 所有持續困難樣本
    all_difficult_path = os.path.join(OUTPUT_DIR, f"all_difficult_samples_{timestamp}.txt")
    with open(all_difficult_path, 'w') as f:
        f.write(f"# 持續困難樣本排除列表 (至少{MIN_EPOCHS_AS_DIFFICULT}個epoch在bin {DIFFICULT_BIN})\n")
        f.write(f"# 生成日期: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# 總數: {len(persistent_difficult_samples)}\n\n")
        for sample_id in sorted(persistent_difficult_samples):
            f.write(f"{sample_id}\n")
    
    print(f"已生成所有持續困難樣本排除列表: {all_difficult_path}")
    
    # 高度困難樣本
    high_difficult_path = os.path.join(OUTPUT_DIR, f"high_difficulty_samples_{timestamp}.txt")
    with open(high_difficult_path, 'w') as f:
        f.write(f"# 高度困難樣本排除列表 (在超過{high_difficulty_threshold}個epoch中在bin {DIFFICULT_BIN})\n")
        f.write(f"# 生成日期: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# 總數: {len(high_difficulty_samples)}\n\n")
        for sample_id in sorted(high_difficulty_samples):
            f.write(f"{sample_id}\n")
    
    print(f"已生成高度困難樣本排除列表: {high_difficult_path}")
    
    # 按角度分類的困難樣本
    for angle, samples in samples_by_angle.items():
        angle_path = os.path.join(OUTPUT_DIR, f"angle_{angle}_difficult_samples_{timestamp}.txt")
        with open(angle_path, 'w') as f:
            f.write(f"# 角度{angle}°的困難樣本排除列表\n")
            f.write(f"# 生成日期: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# 總數: {len(samples)}\n\n")
            for sample_id in sorted(samples):
                f.write(f"{sample_id}\n")
        
        print(f"已生成角度{angle}°的困難樣本排除列表: {angle_path}")

if __name__ == "__main__":
    generate_exclusion_lists() 