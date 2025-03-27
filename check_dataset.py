"""
檢查數據集功能：驗證與視覺化 SpectrogramDatasetWithMaterial 的輸出
功能：
- 檢查資料集載入是否正確
- 顯示數據形狀與統計資訊
- 視覺化頻譜圖樣本
- 驗證標籤分佈
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from datasets import SpectrogramDatasetWithMaterial
from config import DEVICE, DATA_ROOT, CLASSES, MATERIALS, MATERIAL, SEQ_NUMS

def inspect_dataset(dataset, num_samples=5):
    """檢查數據集的基本信息和樣本"""
    print(f"\n{'='*50}")
    print(f"數據集基本信息:")
    print(f"總樣本數: {len(dataset)}")
    
    if len(dataset) == 0:
        print("警告: 數據集為空!")
        return
        
    # 檢查數據形狀，使用try-except避免遞迴錯誤
    try:
        data, label = dataset[0]
        print(f"數據形狀: {data.shape}")
        print(f"標籤類型: {type(label)}")
    except RecursionError:
        print("警告: 存取數據時發生遞迴錯誤")
        return
        
    # 檢查標籤分佈，安全獲取標籤
    try:
        labels = []
        for i in range(len(dataset)):
            try:
                _, label = dataset[i]
                if hasattr(label, 'item'):
                    labels.append(label.item())
                else:
                    labels.append(label)
            except Exception as e:
                print(f"  無法存取樣本 {i} 的標籤: {e}")
        
        if labels:
            unique_labels, counts = np.unique(labels, return_counts=True)
            print("\n標籤分佈:")
            for lbl, cnt in zip(unique_labels, counts):
                print(f"  類別 {CLASSES[int(lbl)]} (標籤 {lbl}): {cnt} 樣本")
        else:
            print("無法獲取標籤資訊")
    except Exception as e:
        print(f"處理標籤時出錯: {e}")
    
    # 避免收集所有數據，這可能造成遞迴
    print("\n檢查個別樣本數據統計量，而非整個數據集:")
    try:
        sample_data, _ = dataset[0]
        print(f"  第一個樣本最小值: {sample_data.min().item():.4f}")
        print(f"  第一個樣本最大值: {sample_data.max().item():.4f}")
    except Exception as e:
        print(f"  無法分析樣本數據: {e}")
    
    # 安全地顯示文件路徑
    if hasattr(dataset, 'paths') and len(dataset.paths) > 0:
        print("\n樣本文件路徑示例:")
        for i in range(min(num_samples, len(dataset.paths))):
            try:
                class_name = "未知"
                try:
                    _, label = dataset[i]
                    if hasattr(label, 'item'):
                        class_name = CLASSES[label.item()]
                    else:
                        class_name = CLASSES[label]
                except:
                    pass
                print(f"  樣本 {i}: 類別 {class_name} | 路徑: {dataset.paths[i]}")
            except Exception as e:
                print(f"  無法顯示樣本 {i} 的路徑: {e}")
    
    return labels, None  # 不返回全部數據，避免內存問題

def visualize_samples(dataset, labels, num_samples=5):
    """視覺化數據集中的樣本"""
    if len(dataset) == 0:
        return
        
    plt.figure(figsize=(15, 10))
    
    # 獲取每個類別的索引
    class_indices = {}
    for i, label in enumerate(labels):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(i)
    
    # 為每個類別選擇一個樣本 (如果可能)
    samples_to_show = []
    for label in sorted(class_indices.keys()):
        if class_indices[label]:
            samples_to_show.append(class_indices[label][0])
            
    # 如果類別數量不夠，從整個數據集中隨機選擇
    if len(samples_to_show) < num_samples:
        import random
        additional = random.sample(range(len(dataset)), 
                                  min(num_samples - len(samples_to_show), len(dataset)))
        samples_to_show.extend(additional)
    
    samples_to_show = samples_to_show[:num_samples]
    
    for i, idx in enumerate(samples_to_show):
        data, label = dataset[idx]
        
        plt.subplot(num_samples, 2, i*2 + 1)
        # 顯示文件路徑
        file_path = dataset.paths[idx] if hasattr(dataset, 'paths') else "路徑未知"
        plt.title(f"樣本 {idx}: 類別 {CLASSES[label.item()]} (標籤 {label.item()})\n{os.path.basename(file_path)}", fontsize=10)
        
        # 顯示頻譜圖
        spect = data.squeeze().numpy()  # 移除通道維度
        plt.imshow(spect, aspect='auto', origin='lower')
        plt.colorbar(format='%+2.0f dB')
        plt.xlabel('時間幀')
        plt.ylabel('頻率箱')
        
        # 顯示波形 (如果dataset有freqs屬性)
        if i == 0 and hasattr(dataset, 'freqs') and dataset.freqs is not None:
            plt.subplot(num_samples, 2, i*2 + 2)
            plt.title(f"頻率響應 - {os.path.basename(file_path)}", fontsize=10)
            # 計算平均頻率響應
            avg_response = np.mean(spect, axis=1)
            plt.plot(dataset.freqs, avg_response)
            plt.xlabel('頻率 (Hz)')
            plt.ylabel('響應 (dB)')
            plt.xscale('log')
            plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "dataset_samples.png"))
    plt.show()

def main():
    """主函數：檢查和視覺化數據集"""
    print("開始檢查數據集...")
    
    selected_freq = '1000hz'
    selected_material = MATERIAL
    
    # 方法1: 按序列號分割
    train_seqs = SEQ_NUMS[:6]  # 使用序列號00-05進行訓練
    test_seqs = SEQ_NUMS[6:]   # 使用序列號06-08進行測試
    
    print(f"Training sequences: {train_seqs}")
    print(f"Testing sequences: {test_seqs}")
    
    # 創建訓練數據集
    try:
        train_dataset = SpectrogramDatasetWithMaterial(
            data_dir=DATA_ROOT,
            classes=CLASSES,
            selected_seqs=train_seqs,
            selected_freq=selected_freq,
            material=selected_material
        )
        
        print("\n檢查訓練數據集:")
        labels, data = inspect_dataset(train_dataset)
        visualize_samples(train_dataset, labels)
        
    except Exception as e:
        print(f"訓練數據集載入錯誤: {e}")
    
    # 創建測試數據集
    try:
        test_dataset = SpectrogramDatasetWithMaterial(
            data_dir=DATA_ROOT,
            classes=CLASSES,
            selected_seqs=test_seqs,
            selected_freq=selected_freq,
            material=selected_material
        )
        
        print("\n檢查測試數據集:")
        labels, data = inspect_dataset(test_dataset)
        visualize_samples(test_dataset, labels)
        
    except Exception as e:
        print(f"測試數據集載入錯誤: {e}")
    
    print("\n數據集檢查完成。")

if __name__ == "__main__":
    main()
