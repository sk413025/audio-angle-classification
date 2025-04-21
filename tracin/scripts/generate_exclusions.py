#!/usr/bin/env python
"""
根據TracIn影響力分數生成樣本排除列表

此腳本分析TracIn計算的影響力分數，識別對模型泛化能力有負面影響的訓練樣本，
並生成排除列表供訓練時使用。
"""

import os
import argparse
import json
from collections import defaultdict
import numpy as np
from pathlib import Path
import datetime
import sys

# 添加父目錄到 Python 路徑，以便在獨立運行時能夠導入模組
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 導入 tracin 模組的功能
from tracin.utils.influence_utils import (
    get_harmful_samples,
    extract_sample_ids,
    save_exclusion_list
)
from tracin.utils.visualization import plot_harmful_samples


def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='生成基於TracIn影響力的樣本排除列表')
    parser.add_argument('--metadata-file', type=str, required=True,
                        help='TracIn影響力元數據文件路徑')
    parser.add_argument('--output-file', type=str, required=True,
                        help='輸出排除列表文件路徑')
    parser.add_argument('--threshold', type=float, default=-5.0,
                        help='負面影響力閾值（低於此值的樣本將被排除）')
    parser.add_argument('--min-occurrences', type=int, default=3,
                        help='一個樣本至少在多少個測試樣本上有負面影響才被排除')
    parser.add_argument('--max-exclusions', type=int, default=50,
                        help='最大排除樣本數量')
    parser.add_argument('--consider-both-samples', action='store_true',
                        help='同時考慮訓練對中的兩個樣本')
    parser.add_argument('--verbose', action='store_true',
                        help='顯示詳細輸出')
    parser.add_argument('--update-metadata', action='store_true',
                        help='更新樣本元數據，記錄排除原因')
    parser.add_argument('--plot-harmful-samples', action='store_true',
                        help='繪製有害樣本的影響力圖表')
    parser.add_argument('--plots-dir', type=str, default='tracin_plots',
                        help='圖表輸出目錄')
    return parser.parse_args()


def update_sample_metadata(samples_to_exclude, metadata_file, verbose):
    """更新樣本元數據，記錄排除原因"""
    # 檢查元數據文件是否存在
    if not os.path.exists(metadata_file):
        print(f"警告: 元數據文件 {metadata_file} 不存在，無法更新樣本元數據")
        return False
    
    try:
        # 加載樣本元數據文件（不是TracIn影響力文件）
        # 從TracIn影響力文件路徑推導樣本元數據文件
        base_path = Path(metadata_file).parent
        material_freq = Path(metadata_file).stem.split('_influence')[0]
        samples_metadata_file = os.path.join(base_path, f"{material_freq}_metadata.json")
        
        if not os.path.exists(samples_metadata_file):
            if verbose:
                print(f"警告: 樣本元數據文件 {samples_metadata_file} 不存在，嘗試創建新文件")
            samples_metadata = {}
        else:
            with open(samples_metadata_file, 'r') as f:
                samples_metadata = json.load(f)
                if verbose:
                    print(f"已加載 {len(samples_metadata)} 個樣本的元數據")
        
        # 更新每個排除樣本的元數據
        updated_count = 0
        for item in samples_to_exclude:
            sample_id = item['sample_id']
            
            # 如果樣本元數據不存在，創建基本記錄
            if sample_id not in samples_metadata:
                # 從樣本ID中提取信息
                parts = sample_id.split('_')
                material = parts[0] if len(parts) > 0 else "unknown"
                cls = parts[1] if len(parts) > 1 else "unknown"
                freq = parts[2] if len(parts) > 2 else "unknown"
                seq_num = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0
                
                # 從類名中提取角度（如果可能）
                angle = 0
                if cls.startswith("deg"):
                    try:
                        angle = float(cls[3:])
                    except ValueError:
                        pass
                
                # 創建新的元數據記錄
                samples_metadata[sample_id] = {
                    "id": sample_id,
                    "file_path": "",  # 無法從ID直接推導完整路徑
                    "angle": angle,
                    "material": material,
                    "frequency": freq,
                    "seq_num": seq_num,
                    "excluded": False,
                    "notes": "",
                    "ghm_bins": {}
                }
            
            # 更新元數據，標記為已排除並添加原因
            metadata = samples_metadata[sample_id]
            metadata['excluded'] = True
            
            # 準備影響參考詳情
            example_influences = []
            for test_id, score in item['examples']:
                example_influences.append(f"{test_id}: {score:.4f}")
            
            # 記錄排除原因
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            metadata['notes'] = (
                f"TracIn分析排除 ({timestamp}): "
                f"此樣本對{item['negative_occurrences']}個測試樣本有負面影響，"
                f"平均影響力分數: {item['average_influence']:.4f}. "
                f"示例: {'; '.join(example_influences)}"
            )
            
            # 添加詳細的TracIn影響信息
            if 'tracin_info' not in metadata:
                metadata['tracin_info'] = {}
            
            metadata['tracin_info']['exclusion_date'] = timestamp
            metadata['tracin_info']['negative_occurrences'] = item['negative_occurrences']
            metadata['tracin_info']['average_influence'] = item['average_influence']
            metadata['tracin_info']['affected_test_samples'] = [test_id for test_id, _ in item['influences']]
            
            # 更新元數據
            samples_metadata[sample_id] = metadata
            updated_count += 1
        
        # 保存更新後的元數據
        with open(samples_metadata_file, 'w') as f:
            json.dump(samples_metadata, f, indent=2)
        
        if verbose:
            print(f"已更新 {updated_count} 個樣本的元數據並保存到 {samples_metadata_file}")
        
        return True
    
    except Exception as e:
        print(f"更新樣本元數據時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_generate_exclusions(args=None):
    """運行排除列表生成"""
    if args is None:
        args = parse_args()
    
    print(f"開始生成基於TracIn影響力的排除列表...")
    
    # 分析影響力元數據
    try:
        harmful_samples = get_harmful_samples(
            args.metadata_file,
            args.threshold,
            args.min_occurrences,
            score_prefix="tracin_influence_"
        )
        
        if not harmful_samples:
            print("沒有找到符合條件的有害樣本。")
            return
        
        if args.verbose:
            print(f"識別出 {len(harmful_samples)} 個潛在有害樣本")
        
        # 生成排除列表
        samples_to_exclude = harmful_samples[:args.max_exclusions]
        
        # 如果consider_both_samples被設置，我們在這裡不處理，直接使用原始training pair
        # 處理樣本對，提取單個樣本
        if args.consider_both_samples and False:  # 禁用此功能，保留原始ID
            # 如果考慮對中的兩個樣本，需要展開樣本對
            expanded_samples = []
            for item in samples_to_exclude:
                sample_ids = extract_sample_ids(item['sample_id'], consider_both=True)
                for sample_id in sample_ids:
                    expanded_samples.append({
                        'sample_id': sample_id,
                        'negative_occurrences': item['negative_occurrences'],
                        'average_influence': item['average_influence'],
                        'examples': item['examples'],
                        'influences': item['influences']
                    })
            samples_to_exclude = expanded_samples
        
        # 保存排除列表
        num_excluded = save_exclusion_list(
            samples_to_exclude,
            args.output_file,
            args.max_exclusions
        )
        
        if args.verbose:
            print(f"已將 {num_excluded} 個樣本寫入排除列表: {args.output_file}")
            print("\n前5個排除樣本及其負面影響:")
            for i, item in enumerate(samples_to_exclude[:5], 1):
                print(f"{i}. {item['sample_id']}")
                print(f"   負面影響出現次數: {item['negative_occurrences']}")
                print(f"   平均影響力分數: {item['average_influence']:.4f}")
                print(f"   示例影響 (測試樣本, 分數):")
                for test_id, score in item['examples']:
                    print(f"   - {test_id}: {score:.4f}")
                print()
        
        # 如果要求更新元數據，記錄排除原因
        if args.update_metadata:
            success = update_sample_metadata(
                samples_to_exclude,
                args.metadata_file,
                args.verbose
            )
            if success:
                print("已成功更新樣本元數據，記錄了排除原因")
            else:
                print("更新樣本元數據失敗")
        
        # 如果需要，繪製有害樣本圖表
        if args.plot_harmful_samples:
            material = Path(args.metadata_file).stem.split('_')[0]
            frequency = Path(args.metadata_file).stem.split('_')[1]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            plots_dir = os.path.join(args.plots_dir, f"{material}_{frequency}")
            os.makedirs(plots_dir, exist_ok=True)
            
            plot_path = os.path.join(plots_dir, f"harmful_samples_{timestamp}.png")
            plot_harmful_samples(
                harmful_samples=samples_to_exclude,
                save_path=plot_path,
                title=f"Harmful Samples for {material}_{frequency}",
                max_samples=min(20, len(samples_to_exclude))
            )
            print(f"有害樣本圖表已保存到: {plot_path}")
        
        print(f"\n排除列表生成完成。可用於訓練時使用 --exclusions-file={args.output_file} 參數排除有害樣本。")
        
        # 顯示使用示例
        material = Path(args.metadata_file).stem.split('_')[0]
        frequency = Path(args.metadata_file).stem.split('_')[1]
        
        print("\n使用示例:")
        print(f"python train.py --frequency {frequency} --material {material} --exclusions-file={args.output_file}")
        
        return samples_to_exclude
        
    except Exception as e:
        print(f"生成排除列表時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函數"""
    run_generate_exclusions()


if __name__ == "__main__":
    main() 