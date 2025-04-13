#!/usr/bin/env python3
"""
清理模型檔案腳本 - 只保留每個 epoch 最新的模型權重
"""

import os
import re
from collections import defaultdict

def cleanup_model_checkpoints(directory):
    """
    清理指定目錄中的模型檢查點，只保留每個 epoch 最新的權重檔案
    
    參數:
        directory: 模型檢查點所在的目錄路徑
    """
    print(f"正在清理目錄: {directory}")
    
    # 檢查目錄是否存在
    if not os.path.exists(directory):
        print(f"錯誤: 目錄 {directory} 不存在")
        return
    
    # 獲取目錄中的所有檔案
    files = os.listdir(directory)
    
    # 模型檔案格式: model_epoch_X_YYYYMMDD_HHMMSS.pt
    pattern = r'model_epoch_(\d+)_(\d{8}_\d{6})\.pt'
    
    # 用字典按 epoch 分組
    epoch_files = defaultdict(list)
    
    # 收集所有符合模式的檔案並分組
    for file in files:
        match = re.match(pattern, file)
        if match:
            epoch = int(match.group(1))
            timestamp = match.group(2)
            epoch_files[epoch].append((timestamp, file))
    
    # 要刪除的檔案列表
    files_to_delete = []
    
    # 對每個 epoch，只保留最新的檔案
    for epoch, file_list in epoch_files.items():
        # 按時間戳排序（由新到舊）
        sorted_files = sorted(file_list, reverse=True)
        
        # 保留最新的檔案，將其他檔案加入到刪除列表
        for i, (_, filename) in enumerate(sorted_files):
            if i > 0:  # 第一個是最新的，跳過
                files_to_delete.append(filename)
    
    # 顯示將被刪除的檔案
    if files_to_delete:
        print(f"將刪除以下 {len(files_to_delete)} 個檔案:")
        for file in files_to_delete:
            print(f"  - {file}")
        
        # 詢問用戶確認
        confirm = input("確定要刪除這些檔案嗎? (y/n): ")
        if confirm.lower() == 'y':
            # 刪除檔案
            for file in files_to_delete:
                file_path = os.path.join(directory, file)
                try:
                    os.remove(file_path)
                    print(f"已刪除: {file}")
                except Exception as e:
                    print(f"刪除 {file} 時出錯: {e}")
            print("清理完成!")
        else:
            print("操作已取消")
    else:
        print("沒有找到需要清理的檔案")

if __name__ == "__main__":
    # 指定要清理的目錄
    directory = "./saved_models/model_checkpoints/plastic_3000hz"
    cleanup_model_checkpoints(directory) 