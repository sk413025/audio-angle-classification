#!/usr/bin/env python
"""
比較兩個批次一致性檢查文件，確認它們是否相同
用法: python compare_batch_consistency.py [file1.pkl] [file2.pkl]
"""

import os
import sys
import pickle
import numpy as np
from glob import glob
import argparse


def load_pickle_file(file_path):
    """載入pickle檔案"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"載入檔案 {file_path} 時發生錯誤: {e}")
        return None


def compare_batch_data(batch1, batch2, tolerance=1e-6):
    """
    比較兩個批次資料是否相同
    
    參數:
        batch1: 第一個批次資料
        batch2: 第二個批次資料
        tolerance: 浮點數比較的容忍度
        
    返回:
        相同: True/False
        差異: 差異列表
    """
    differences = []
    
    # 檢查批次索引
    if batch1['batch_idx'] != batch2['batch_idx']:
        differences.append(f"批次索引不同: {batch1['batch_idx']} vs {batch2['batch_idx']}")
    
    # 檢查數據形狀
    if batch1['data1_shape'] != batch2['data1_shape']:
        differences.append(f"Data1形狀不同: {batch1['data1_shape']} vs {batch2['data1_shape']}")
    
    if batch1['data2_shape'] != batch2['data2_shape']:
        differences.append(f"Data2形狀不同: {batch1['data2_shape']} vs {batch2['data2_shape']}")
    
    # 檢查標籤
    if batch1['targets'] != batch2['targets']:
        differences.append(f"目標值不同: {batch1['targets']} vs {batch2['targets']}")
    
    if batch1['labels1'] != batch2['labels1']:
        differences.append(f"標籤1不同: {batch1['labels1']} vs {batch2['labels1']}")
    
    if batch1['labels2'] != batch2['labels2']:
        differences.append(f"標籤2不同: {batch1['labels2']} vs {batch2['labels2']}")
    
    # 檢查數據統計特徵
    if abs(batch1['data1_sum'] - batch2['data1_sum']) > tolerance:
        differences.append(f"Data1總和不同: {batch1['data1_sum']} vs {batch2['data1_sum']}")
    
    if abs(batch1['data2_sum'] - batch2['data2_sum']) > tolerance:
        differences.append(f"Data2總和不同: {batch1['data2_sum']} vs {batch2['data2_sum']}")
    
    if abs(batch1['data1_std'] - batch2['data1_std']) > tolerance:
        differences.append(f"Data1標準差不同: {batch1['data1_std']} vs {batch2['data1_std']}")
    
    if abs(batch1['data2_std'] - batch2['data2_std']) > tolerance:
        differences.append(f"Data2標準差不同: {batch1['data2_std']} vs {batch2['data2_std']}")
    
    return len(differences) == 0, differences


def compare_batch_files(file1, file2, tolerance=1e-6):
    """
    比較兩個批次檔案
    
    參數:
        file1: 第一個檔案路徑
        file2: 第二個檔案路徑
        tolerance: 浮點數比較的容忍度
        
    返回:
        完全相同: True/False
    """
    data1 = load_pickle_file(file1)
    data2 = load_pickle_file(file2)
    
    if data1 is None or data2 is None:
        return False
    
    if len(data1) != len(data2):
        print(f"批次數量不同: 檔案1有 {len(data1)} 批次，檔案2有 {len(data2)} 批次")
        return False
    
    all_identical = True
    total_differences = 0
    
    for i, (batch1, batch2) in enumerate(zip(data1, data2)):
        identical, differences = compare_batch_data(batch1, batch2, tolerance)
        if not identical:
            all_identical = False
            total_differences += len(differences)
            print(f"\n批次 {i} 有差異:")
            for diff in differences:
                print(f"  - {diff}")
    
    return all_identical, total_differences


def find_latest_pkl_files(directory, n=2):
    """找到指定目錄下最新的n個pkl文件"""
    pkl_files = glob(os.path.join(directory, '*.pkl'))
    return sorted(pkl_files, key=os.path.getmtime, reverse=True)[:n]


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='比較兩個批次一致性檢查文件')
    parser.add_argument('files', nargs='*', help='要比較的pickle檔案')
    parser.add_argument('--dir', '-d', help='自動尋找目錄中最新的兩個pkl檔案')
    parser.add_argument('--tolerance', '-t', type=float, default=1e-6, 
                        help='浮點數比較的容忍度 (預設: 1e-6)')
    args = parser.parse_args()
    
    files_to_compare = []
    
    # 如果指定了目錄，找到最新的兩個pkl檔案
    if args.dir:
        if os.path.isdir(args.dir):
            files_to_compare = find_latest_pkl_files(args.dir)
            if len(files_to_compare) < 2:
                print(f"在目錄 {args.dir} 中找不到足夠的pkl檔案")
                return
        else:
            print(f"目錄 {args.dir} 不存在")
            return
    # 否則使用命令行參數指定的檔案
    elif len(args.files) >= 2:
        files_to_compare = args.files[:2]
    else:
        print("請提供兩個pkl檔案進行比較或指定一個包含pkl檔案的目錄")
        return
    
    print(f"比較檔案:\n1. {files_to_compare[0]}\n2. {files_to_compare[1]}")
    identical, total_differences = compare_batch_files(
        files_to_compare[0], files_to_compare[1], args.tolerance
    )
    
    if identical:
        print("\n結果: 批次資料完全相同！這表示每次訓練的批次順序和內容都是固定的。")
    else:
        print(f"\n結果: 批次資料不同！總共發現 {total_differences} 個差異。")
        print("這表示每次訓練的批次順序或內容有所變化。")


if __name__ == "__main__":
    main() 