"""
TracIn 影響力處理工具

這個模組提供處理 TracIn 影響力分數的實用函數。
"""

import os
import json
from typing import Dict, List, Union, Tuple, Optional, Any
from pathlib import Path
import numpy as np


def load_influence_scores(metadata_file: str, score_name: str = "tracin_influence") -> Dict[str, float]:
    """
    從元數據文件中加載影響力分數。
    
    Args:
        metadata_file: 影響力元數據文件路徑
        score_name: 要加載的影響力分數名稱
        
    Returns:
        字典，鍵為樣本 ID，值為影響力分數
    """
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"找不到元數據文件: {metadata_file}")
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f"元數據文件 {metadata_file} 不是有效的 JSON 格式")
    
    # 提取影響力分數
    influence_scores = {}
    for sample_id, sample_data in metadata.items():
        for key, value in sample_data.items():
            if key.startswith(score_name):
                influence_scores[sample_id] = value
                break
    
    return influence_scores


def get_harmful_samples(
    metadata_file: str,
    threshold: float = -5.0,
    min_occurrences: int = 3,
    score_prefix: str = "tracin_influence_"
) -> List[Dict[str, Any]]:
    """
    從元數據文件中識別有害樣本。
    
    Args:
        metadata_file: 影響力元數據文件路徑
        threshold: 負面影響力閾值，低於此值的影響力被視為負面
        min_occurrences: 樣本至少在多少個測試樣本上有負面影響才被視為有害
        score_prefix: 影響力分數的前綴
        
    Returns:
        有害樣本列表，每個樣本為一個字典，包含以下字段：
        - sample_id: 樣本 ID (training pair)
        - negative_occurrences: 負面影響出現次數
        - average_influence: 平均影響力分數
        - examples: 示例影響（測試樣本對 ID 和影響力分數）
    """
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"找不到元數據文件: {metadata_file}")
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f"元數據文件 {metadata_file} 不是有效的 JSON 格式")
    
    # 收集每個 training pair 的負面影響
    pair_negative_influences = {}
    
    for train_pair_id, test_pairs in metadata.items():
        negative_influences = []
        
        for test_key, score in test_pairs.items():
            if test_key.startswith(score_prefix) and score < threshold:
                # 從測試鍵中提取測試對 ID
                test_id = test_key[len(score_prefix):]
                negative_influences.append((test_id, score))
        
        if len(negative_influences) >= min_occurrences:
            avg_influence = sum(score for _, score in negative_influences) / len(negative_influences)
            
            # 直接使用原始的training pair ID
            pair_negative_influences[train_pair_id] = {
                'sample_id': train_pair_id,
                'negative_occurrences': len(negative_influences),
                'average_influence': avg_influence,
                'examples': negative_influences[:3],  # 保存前 3 個例子
                'influences': negative_influences     # 所有影響
            }
    
    # 轉換為列表並排序
    harmful_samples = list(pair_negative_influences.values())
    harmful_samples.sort(key=lambda x: (x['negative_occurrences'], x['average_influence']), reverse=True)
    
    return harmful_samples


def extract_sample_ids(pair_id: str, consider_both: bool = True) -> List[str]:
    """
    從樣本對 ID 中提取單個樣本 ID。
    
    此函數支援多種類型的樣本對ID格式：
    1. 完整排序對格式：如 "material_degXXX_freq_seq_material_degYYY_freq_seq"
       例如 "plastic_deg000_500hz_01_plastic_deg090_500hz_02"
    2. 部分排序對格式：如 "material_degXXX_freq_material_degYYY_freq"
       例如 "plastic_deg000_500hz_plastic_deg090_500hz"
    3. 單樣本 ID（完整）：如 "material_degXXX_freq_seq"
       例如 "plastic_deg090_500hz_02"
    4. 單樣本 ID（部分）：如 "degXXX_freq_seq" 或 "degXXX_freq"
       例如 "deg090_500hz_02" 或 "deg090_500hz"
    
    Args:
        pair_id: 樣本對 ID
        consider_both: 是否同時考慮樣本對中的兩個樣本
        
    Returns:
        樣本 ID 列表，通常包含1-2個元素
    """
    # 將 ID 按下劃線分割
    parts = pair_id.split('_')
    
    # 查找包含角度信息的部分（通常以 "deg" 開頭）
    deg_indices = [i for i, part in enumerate(parts) if part.startswith('deg')]
    
    # 如果找到至少兩個角度信息，說明這是一個排序對 ID
    if len(deg_indices) >= 2:
        # 找到第二個樣本的起始位置
        mid_point = deg_indices[1]
        
        # 標準化第一個樣本ID（確保包含材料、角度、頻率和序列號）
        # 這裡我們假設第一個樣本至少有角度和頻率
        sample1_parts = parts[:mid_point]
        
        # 檢查第一個樣本的格式
        # 如果需要，添加缺少的序列號
        if len(sample1_parts) >= 3 and sample1_parts[-1].startswith("hz"):
            # 只有角度和頻率，沒有序列號，添加一個默認的序列號 "00"
            if consider_both:
                sample1_parts.append("00")
        
        # 標準化第二個樣本ID
        sample2_parts = parts[mid_point:]
        
        # 如果第二個樣本只包含 deg*_freq，需要添加材料和序列號
        if len(sample2_parts) == 2 and sample2_parts[0].startswith("deg") and sample2_parts[1].startswith("hz"):
            # 添加材料前綴（假設與第一個樣本相同）
            if "plastic" in sample1_parts or "metal" in sample1_parts or "wood" in sample1_parts:
                for part in sample1_parts:
                    if part in ["plastic", "metal", "wood"]:
                        sample2_parts.insert(0, part)
                        break
            
            # 添加默認序列號
            if consider_both:
                sample2_parts.append("00")
        
        # 生成標準化的樣本 ID
        sample1_id = '_'.join(sample1_parts)
        sample2_id = '_'.join(sample2_parts)
        
        # 如果樣本 ID 仍然不完整或不符合預期格式，嘗試進一步修正
        # 檢查第二個樣本是否以 deg 開頭但缺少材料前綴
        if sample2_id.startswith("deg") and "plastic" in sample1_id:
            sample2_id = f"plastic_{sample2_id}"
        
        if consider_both:
            return [sample1_id, sample2_id]
        else:
            return [sample1_id]  # 只返回第一個樣本
    
    # 如果只找到一個角度信息，可能是單個樣本ID
    elif len(deg_indices) == 1:
        # 將缺失的材料前綴添加到單個樣本ID中
        if pair_id.startswith("deg"):
            corrected_id = f"plastic_{pair_id}"
            return [corrected_id]
    
    # 如果無法識別為排序對或單個樣本，返回完整ID作為單一樣本
    return [pair_id]


def save_exclusion_list(harmful_samples: List[Dict[str, Any]], output_file: str, max_exclusions: int = 50) -> int:
    """
    保存排除列表到文件。
    
    Args:
        harmful_samples: 有害樣本列表
        output_file: 輸出文件路徑
        max_exclusions: 最大排除樣本數量
        
    Returns:
        保存的樣本數量
    """
    # 限制總的排除樣本數量
    samples_to_exclude = harmful_samples[:max_exclusions]
    
    # 確保輸出目錄存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w') as f:
        # 添加頭部註釋
        f.write("# Pair exclusion list generated by TracIn analysis\n")
        f.write(f"# Generated on: {import_datetime().datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("# Format: One pair per line in format sample1_sample2\n\n")
        
        # 直接使用原始的training pair ID
        pairs_written = 0
        for item in samples_to_exclude:
            if pairs_written >= max_exclusions:
                break
                
            # 直接寫入原始sample_id，這是training pair
            sample_id = item['sample_id']
            f.write(f"{sample_id}\n")
            pairs_written += 1
    
    return pairs_written


def import_datetime():
    """導入 datetime 模組"""
    import datetime
    return datetime 