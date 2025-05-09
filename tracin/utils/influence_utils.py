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
        - sample_id: 樣本 ID
        - negative_occurrences: 負面影響出現次數
        - average_influence: 平均影響力分數
        - examples: 示例影響（測試樣本 ID 和影響力分數）
    """
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"找不到元數據文件: {metadata_file}")
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f"元數據文件 {metadata_file} 不是有效的 JSON 格式")
    
    # 收集每個樣本的負面影響
    sample_negative_influences = {}
    
    for sample_id, sample_data in metadata.items():
        negative_influences = []
        
        for key, value in sample_data.items():
            if key.startswith(score_prefix) and value < threshold:
                # 從鍵中提取測試樣本 ID
                test_id = key[len(score_prefix):]
                negative_influences.append((test_id, value))
        
        if len(negative_influences) >= min_occurrences:
            avg_influence = sum(score for _, score in negative_influences) / len(negative_influences)
            sample_negative_influences[sample_id] = {
                'sample_id': sample_id,
                'negative_occurrences': len(negative_influences),
                'average_influence': avg_influence,
                'examples': negative_influences[:3],  # 保存前 3 個例子
                'influences': negative_influences     # 所有影響
            }
    
    # 轉換為列表並排序
    harmful_samples = list(sample_negative_influences.values())
    harmful_samples.sort(key=lambda x: (x['negative_occurrences'], x['average_influence']), reverse=True)
    
    return harmful_samples


def extract_sample_ids(pair_id: str, consider_both: bool = True) -> List[str]:
    """
    從樣本對 ID 中提取單個樣本 ID。
    
    Args:
        pair_id: 樣本對 ID，格式為 "sample1_sample2"
        consider_both: 是否同時考慮樣本對中的兩個樣本
        
    Returns:
        樣本 ID 列表
    """
    parts = pair_id.split('_')
    
    # 找到包含度數信息的部分
    deg_indices = [i for i, part in enumerate(parts) if part.startswith('deg')]
    if len(deg_indices) < 2:
        return []
    
    # 找到第一個樣本 ID 的結束位置和第二個樣本 ID 的開始位置
    if len(deg_indices) >= 2:
        mid_point = deg_indices[1]
        sample1_parts = parts[:mid_point+2]  # +2 to include the degree number and the sequence number
        sample2_parts = parts[mid_point:]
        
        sample1_id = '_'.join(sample1_parts)
        sample2_id = '_'.join(sample2_parts)
        
        if consider_both:
            return [sample1_id, sample2_id]
        else:
            return [sample1_id]  # 只返回第一個樣本
    
    return []


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
    # 限制排除樣本數量
    samples_to_exclude = harmful_samples[:max_exclusions]
    
    # 提取樣本 ID
    exclusion_list = [item['sample_id'] for item in samples_to_exclude]
    
    # 確保輸出目錄存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 寫入排除列表
    with open(output_file, 'w') as f:
        # 添加頭部註釋
        f.write("# Sample exclusion list generated by TracIn analysis\n")
        f.write(f"# Generated on: {import_datetime().datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("# Format: One sample ID per line\n\n")
        
        for sample_id in exclusion_list:
            f.write(f"{sample_id}\n")
    
    return len(exclusion_list)


def import_datetime():
    """導入 datetime 模組"""
    import datetime
    return datetime 