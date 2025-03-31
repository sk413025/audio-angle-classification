"""
視覺化模組：簡化版 - 使用文本檔案而非圖表輸出以避免matplotlib兼容性問題
"""

import os
import numpy as np
import torch
from angle_classification_deg6.config import CLASSES, DEVICE
import csv

def visualize_frequency_weights(model, freqs, save_path):
    """儲存頻率權重到CSV文件而非繪圖"""
    try:
        print("Saving frequency weights data...")
        weights = model.get_frequency_weights()
        
        # 安全檢查
        if freqs is None or len(freqs) == 0:
            print("Warning: Frequency data is empty, skipping visualization")
            return None
            
        # 處理維度不匹配問題
        if len(freqs) != len(weights):
            print(f"Interpolating weights from shape {weights.shape} to match frequencies shape {freqs.shape}")
            
            # 使用最簡單的插值方法
            weights_np = weights.detach().cpu().numpy()
            x_orig = np.linspace(0, 1, len(weights_np))
            x_new = np.linspace(0, 1, len(freqs))
            weights_for_plot = np.interp(x_new, x_orig, weights_np)
        else:
            weights_for_plot = weights.detach().cpu().numpy()
        
        # 安全檢查
        weights_for_plot = np.nan_to_num(weights_for_plot, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 將數據保存到CSV文件
        csv_path = save_path.replace('.png', '.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frequency (Hz)', 'Weight'])
            for freq, weight in zip(freqs, weights_for_plot):
                writer.writerow([freq, weight])
        
        print(f"Frequency weights data saved to {csv_path}")
        
        # 找出重要頻率段
        threshold = np.mean(weights_for_plot) + np.std(weights_for_plot)
        important_mask = weights_for_plot > threshold
        important_freqs = freqs[important_mask] if np.any(important_mask) else []
        
        # 保存重要頻率資訊到文本文件
        txt_path = save_path.replace('.png', '_important_freqs.txt')
        with open(txt_path, 'w') as f:
            f.write("Important Frequency Ranges\n")
            f.write("=======================\n\n")
            
            if len(important_freqs) > 0:
                start_freq = important_freqs[0]
                last_freq = start_freq
                
                for i in range(1, len(important_freqs)):
                    curr_freq = important_freqs[i]
                    if curr_freq - last_freq > (freqs[1] - freqs[0]) * 2:  # 超過兩個頻率間隔視為不連續
                        f.write(f"{start_freq:.1f} Hz - {last_freq:.1f} Hz\n")
                        start_freq = curr_freq
                    last_freq = curr_freq
                
                f.write(f"{start_freq:.1f} Hz - {last_freq:.1f} Hz\n")
                print(f"Important frequency ranges saved to {txt_path}")
            else:
                f.write("No significant frequency ranges found\n")
        
        return weights_for_plot
    
    except Exception as e:
        print(f"Error in visualize_frequency_weights: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def plot_ranking_stats(epochs, train_losses, test_losses, train_accs, test_accs, 
                      save_dir, freq):
    """將訓練統計數據保存為CSV文件"""
    try:
        print("Saving training statistics data...")
        
        # 保存損失和準確率數據到CSV
        csv_path = os.path.join(save_dir, f'training_stats_{freq}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Loss', 'Test Loss', 'Train Acc', 'Test Acc'])
            for i in range(len(epochs)):
                writer.writerow([epochs[i], train_losses[i], test_losses[i], 
                                train_accs[i], test_accs[i]])
        
        print(f"Training statistics saved to {csv_path}")
    
    except Exception as e:
        print(f"Error in plot_ranking_stats: {str(e)}")
        import traceback
        traceback.print_exc()

def save_ranking_results(model, test_dataset, device, save_dir, selected_freq):
    """保存模型排序結果"""
    try:
        print("Saving ranking results...")
        model.eval()
        
        # 計算每個類別的平均排序分數
        class_scores = {cls: [] for cls in CLASSES}
        
        # 產生隨機索引順序，而不是按順序遍歷
        import random
        random_indices = list(range(len(test_dataset)))
        random.shuffle(random_indices)
        
        with torch.no_grad():
            for idx in random_indices:  # 使用隨機順序遍歷數據集
                data, label = test_dataset[idx]
                data = data.unsqueeze(0).to(device)
                score = model(data).item()
                label_idx = label.item() if isinstance(label, torch.Tensor) else label
                if label_idx < len(CLASSES):
                    class_name = CLASSES[label_idx]
                    class_scores[class_name].append(score)
        
        # 計算每個類別的平均分數和標準差
        class_stats = {}
        for cls, scores in class_scores.items():
            if scores:
                class_stats[cls] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'count': len(scores)
                }
        
        # 保存文本結果
        with open(os.path.join(save_dir, f'ranking_results_{selected_freq}.txt'), 'w') as f:
            f.write("Ranking Results By Class\n")
            f.write("===================\n\n")
            f.write("Class | Mean Score | Std Dev | Sample Count\n")
            f.write("-----------------------------------------\n")
            
            for cls in sorted(class_stats.keys()):
                stats = class_stats[cls]
                f.write(f"{cls} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['count']}\n")
        
        # 保存CSV結果
        with open(os.path.join(save_dir, f'ranking_results_{selected_freq}.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Class', 'Mean Score', 'Std Dev', 'Sample Count'])
            for cls in sorted(class_stats.keys()):
                stats = class_stats[cls]
                writer.writerow([cls, stats['mean'], stats['std'], stats['count']])
        
        # 保存模型
        model_path = os.path.join(save_dir, f'model_{selected_freq}.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        print(f"Ranking results saved to {save_dir}")
    
    except Exception as e:
        print(f"Error in save_ranking_results: {str(e)}")
        import traceback
        traceback.print_exc()

def visualize_attention_effect(model, input_spectrogram, save_path):
    """
    跳過注意力效果可視化，對比保存原始數據
    """
    print("Visualization skipped - saving raw data instead")
    
    try:
        # 保存原始頻譜數據到文件
        if isinstance(input_spectrogram, torch.Tensor):
            spec_data = input_spectrogram.detach().cpu().numpy()
            data_path = save_path.replace('.png', '_raw_data.npy')
            np.save(data_path, spec_data)
            print(f"Raw spectrogram data saved to {data_path}")
    except Exception as e:
        print(f"Error saving spectrogram data: {str(e)}")
        import traceback
        traceback.print_exc()
