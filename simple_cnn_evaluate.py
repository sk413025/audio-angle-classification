"""
簡單CNN模型評估腳本
功能：
- 載入訓練好的CNN模型
- 評估模型在測試數據上的性能
- 生成分類報告與混淆矩陣
- 可視化結果
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib
# 設置後端為Agg (非互動式)，避免GUI相關錯誤
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

from datasets import SpectrogramDatasetWithMaterial
from simple_cnn_models import SimpleCNNAudioRanker
import config

def evaluate_cnn_model(model_path, frequency, material, selected_seqs=None, plot_results=True):
    """
    評估訓練好的CNN模型

    參數:
        model_path: 模型權重文件路徑
        frequency: 使用的頻率數據
        material: 使用的材質
        selected_seqs: 選擇的測試序列編號（若為None則使用所有序列）
        plot_results: 是否繪製結果圖表
    """
    # 設置測試序列
    if selected_seqs is None:
        # 使用最後兩個序列作為測試集
        selected_seqs = config.SEQ_NUMS[-2:]
    
    print(f"評估CNN模型 - 頻率: {frequency}, 材質: {material}")
    print(f"使用測試序列: {selected_seqs}")
    
    # 設置裝置
    device = config.DEVICE
    print(f"使用裝置: {device}")
    
    # 加載測試數據
    dataset = SpectrogramDatasetWithMaterial(
        config.DATA_ROOT,
        config.CLASSES,
        selected_seqs,
        frequency,
        material
    )
    
    if len(dataset) == 0:
        print("測試數據集為空，無法評估模型")
        return
    
    # 加載模型
    model = SimpleCNNAudioRanker(n_freqs=dataset.data.shape[2] if dataset.data is not None else None)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"已加載模型權重: {model_path}")
    
    # 進行預測
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            data, label = dataset[i]
            data = data.unsqueeze(0).to(device)  # 添加批次維度
            output = model(data)
            # 確保輸出是一個標量
            output = output.squeeze()
            all_preds.append(output.cpu().numpy())
            all_labels.append(label.item())
    
    # 轉換為 numpy 數組
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels)
    
    # 按排名預測分類（將相對排序轉換為絕對類別）
    # 首先獲取每個類別的平均排序分數
    class_scores = {}
    for cls in range(len(config.CLASSES)):
        indices = np.where(all_labels == cls)[0]
        if len(indices) > 0:
            class_scores[cls] = np.mean(all_preds[indices])
    
    # 對每個樣本，選擇與其排序分數最接近的類別作為預測結果
    predicted_labels = []
    for score in all_preds:
        closest_class = min(class_scores.keys(), key=lambda c: abs(class_scores[c] - score))
        predicted_labels.append(closest_class)
    
    predicted_labels = np.array(predicted_labels)
    
    # 計算準確率
    accuracy = accuracy_score(all_labels, predicted_labels)
    print(f"\n準確率: {accuracy*100:.2f}%")
    
    # 生成分類報告
    class_names = config.CLASSES
    print("\n分類報告:")
    report = classification_report(all_labels, predicted_labels, target_names=class_names)
    print(report)
    
    # 計算混淆矩陣
    cm = confusion_matrix(all_labels, predicted_labels)
    
    # 繪製結果 - 使用try-except捕獲任何可能的繪圖錯誤
    if plot_results:
        try:
            plt.figure(figsize=(12, 5))
            
            # 繪製混淆矩陣
            plt.subplot(1, 2, 1)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('預測標籤')
            plt.ylabel('真實標籤')
            plt.title('混淆矩陣')
            
            # 繪製類別分數分布
            plt.subplot(1, 2, 2)
            for cls in range(len(class_names)):
                indices = np.where(all_labels == cls)[0]
                if len(indices) > 0:
                    plt.scatter(all_preds[indices], [cls] * len(indices), alpha=0.5, label=f"{class_names[cls]}")
            
            # 標記出每個類別的平均分數
            for cls, score in class_scores.items():
                plt.axvline(x=score, color=f"C{cls}", linestyle='--', alpha=0.7)
                
            plt.yticks(range(len(class_names)), class_names)
            plt.xlabel('模型輸出分數')
            plt.ylabel('類別')
            plt.title('分數分布與類別關係')
            plt.legend()
            
            plt.tight_layout()
            
            # 保存圖像
            results_dir = os.path.join(config.SAVE_DIR, "results")
            os.makedirs(results_dir, exist_ok=True)
            output_file = os.path.join(results_dir, f"cnn_eval_{material}_{frequency}.png")
            plt.savefig(output_file)
            print(f"評估結果圖表已保存到: {output_file}")
            
            # 在某些環境中，plt.show()可能會導致問題，所以改為非互動式保存
            # plt.show()
            
        except Exception as e:
            print(f"警告: 繪圖時發生錯誤: {str(e)}")
            print("評估仍然完成，但無法生成視覺化結果。")
    
    return {
        "accuracy": accuracy,
        "predictions": predicted_labels,
        "true_labels": all_labels,
        "scores": all_preds,
        "confusion_matrix": cm
    }

if __name__ == "__main__":
    # 輸入模型路徑
    model_path = input("請輸入模型路徑: ")
    if not os.path.exists(model_path):
        print(f"找不到模型文件: {model_path}")
    else:
        # 從檔名解析材質和頻率
        filename = os.path.basename(model_path)
        parts = filename.split('_')
        if len(parts) >= 3:
            material = parts[1]
            frequency = parts[2]
            evaluate_cnn_model(model_path, frequency, material)
        else:
            # 手動輸入材質和頻率
            material = input("請輸入材質 (box/plastic): ")
            frequency = input("請輸入頻率 (500hz/1000hz/3000hz): ")
            evaluate_cnn_model(model_path, frequency, material)
