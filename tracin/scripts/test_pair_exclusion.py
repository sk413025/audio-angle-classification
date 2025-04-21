#!/usr/bin/env python
"""
測試 ranking pair 排除功能的腳本

此腳本分析 TracIn 影響力元數據，評估不同閾值下可能排除的有害 ranking pair 數量，
並顯示有害 pair 的示例和它們對測試樣本的負面影響。
"""

import os
import sys
import argparse
from pathlib import Path

# 添加父目錄到 Python 路徑，以便在獨立運行時能夠導入模組
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tracin.utils.influence_utils import get_harmful_samples


def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='測試 ranking pair 排除功能')
    
    parser.add_argument('--metadata-file', type=str, required=True,
                        help='TracIn 影響力元數據文件路徑')
    
    parser.add_argument('--output-file', type=str, default=None,
                        help='輸出排除列表文件路徑（可選）')
    
    parser.add_argument('--thresholds', type=str, default="-5.0,-10.0,-15.0",
                        help='測試的閾值列表，用逗號分隔（默認: -5.0,-10.0,-15.0）')
    
    parser.add_argument('--min-occurrences', type=int, default=3,
                        help='最小負面影響出現次數（默認: 3）')
    
    return parser.parse_args()


def main():
    """主函數"""
    args = parse_args()
    
    # 檢查元數據文件是否存在
    if not os.path.exists(args.metadata_file):
        print(f"錯誤: 找不到元數據文件: {args.metadata_file}")
        return 1
    
    print(f"分析 TracIn 影響力元數據: {args.metadata_file}")
    print(f"最小負面影響出現次數: {args.min_occurrences}")
    
    # 解析閾值
    try:
        thresholds = [float(t) for t in args.thresholds.split(',')]
    except ValueError:
        print(f"錯誤: 閾值格式無效: {args.thresholds}")
        return 1
    
    # 從文件名中提取材料和頻率
    file_basename = Path(args.metadata_file).stem
    parts = file_basename.split('_')
    material = parts[0] if len(parts) > 0 else "unknown"
    frequency = parts[1] if len(parts) > 1 else "unknown"
    
    print(f"材料: {material}, 頻率: {frequency}")
    print("\n測試不同閾值下有害 pair 的識別情況:")
    
    # 測試不同閾值
    results = {}
    for threshold in thresholds:
        harmful_samples = get_harmful_samples(
            args.metadata_file,
            threshold=threshold,
            min_occurrences=args.min_occurrences,
            score_prefix="tracin_influence_"
        )
        
        results[threshold] = harmful_samples
        print(f"  閾值 {threshold}: 識別出 {len(harmful_samples)} 個有害 pair")
    
    # 顯示有害 pair 的示例
    print(f"\n有害 pair 示例（使用閾值 {thresholds[0]}）:")
    if results[thresholds[0]]:
        for i, item in enumerate(results[thresholds[0]][:5], 1):
            print(f"{i}. Pair ID: {item['sample_id']}")
            print(f"   負面影響: 出現 {item['negative_occurrences']} 次")
            print(f"   平均影響力分數: {item['average_influence']:.4f}")
            print(f"   對測試樣本的影響示例:")
            for test_id, score in item['examples']:
                print(f"     - {test_id}: {score:.4f}")
            print()
    else:
        print("未找到符合條件的有害 pair。")
    
    # 如果指定了輸出文件，則生成排除列表
    if args.output_file:
        from tracin.scripts.generate_exclusions import run_generate_exclusions
        
        # 創建臨時參數對象用於 generate_exclusions
        from argparse import Namespace
        gen_args = Namespace(
            metadata_file=args.metadata_file,
            output_file=args.output_file,
            threshold=thresholds[0],
            min_occurrences=args.min_occurrences,
            max_exclusions=50,
            pair_mode='ranking_pair',
            verbose=True,
            update_metadata=False,
            plot_harmful_samples=False,
            plots_dir='tracin_plots'
        )
        
        print(f"\n生成排除列表到: {args.output_file}")
        run_generate_exclusions(gen_args)
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 