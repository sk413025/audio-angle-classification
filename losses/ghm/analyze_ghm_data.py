#!/usr/bin/env python
"""
分析元數據中的GHM bin分配
"""

import json
import os
import sys
from collections import defaultdict

# 元數據文件路徑
METADATA_FILE = "/Users/sbplab/Hank/sinetone_sliced/step_018_sliced/metadata/metadata_v1.0.json"

def analyze_ghm_bins():
    """分析並顯示GHM bin分配的統計信息"""
    if not os.path.exists(METADATA_FILE):
        print(f"元數據文件不存在: {METADATA_FILE}")
        return

    # 加載元數據
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    
    print(f"元數據中的樣本總數: {len(metadata)}")
    
    # 統計每個epoch中各bin的樣本數量
    bins_per_epoch = defaultdict(lambda: defaultdict(int))
    samples_with_bins = 0
    
    for sample_id, info in metadata.items():
        ghm_bins = info.get('ghm_bins', {})
        if ghm_bins:
            samples_with_bins += 1
            for epoch, bin_idx in ghm_bins.items():
                bins_per_epoch[epoch][bin_idx] += 1
    
    print(f"有GHM bin分配記錄的樣本數量: {samples_with_bins}")
    
    # 打印每個epoch的bin分配統計
    print("\nGHM bin分配統計:")
    for epoch in sorted(bins_per_epoch.keys()):
        print(f"\nEpoch {epoch}:")
        total_in_epoch = sum(bins_per_epoch[epoch].values())
        
        for bin_idx in sorted(bins_per_epoch[epoch].keys()):
            count = bins_per_epoch[epoch][bin_idx]
            percentage = (count / total_in_epoch) * 100 if total_in_epoch > 0 else 0
            print(f"  Bin {bin_idx}: {count} samples ({percentage:.1f}%)")

if __name__ == "__main__":
    analyze_ghm_bins() 