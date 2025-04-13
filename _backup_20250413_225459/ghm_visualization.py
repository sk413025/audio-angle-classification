"""
GHM 訓練結果視覺化分析腳本
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from matplotlib.ticker import MaxNLocator

def visualize_gradient_density_evolution(stats_dir, save_dir='./ghm_visualizations'):
    """
    視覺化梯度密度分佈的演變過程
    
    參數:
        stats_dir: 存放 GHM 統計數據的目錄
        save_dir: 保存視覺化結果的目錄
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 尋找所有批次0的統計數據，按輪數排序
    batch0_files = sorted(glob.glob(os.path.join(stats_dir, 'epoch*_batch0_*.npy')), 
                         key=lambda x: int(x.split('epoch')[1].split('_')[0]))
    
    if len(batch0_files) == 0:
        print(f"未找到 GHM 統計數據文件在 {stats_dir}")
        return
    
    # 創建梯度密度演變圖
    plt.figure(figsize=(14, 10))
    
    # 選擇幾個關鍵輪數進行可視化
    num_epochs = len(batch0_files)
    sample_indices = np.linspace(0, num_epochs-1, 5, dtype=int)
    
    for i, idx in enumerate(sample_indices):
        file_path = batch0_files[idx]
        epoch_num = int(file_path.split('epoch')[1].split('_')[0])
        stats = np.load(file_path, allow_pickle=True).item()
        
        bin_edges = stats['bin_edges']
        bin_counts = stats['bin_counts']
        
        # 計算每個 bin 的密度
        bin_width = 1.0 / len(bin_counts)
        density = np.array(bin_counts) / sum(bin_counts) / bin_width
        
        # 繪製密度曲線
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.plot(bin_centers, density, '-o', linewidth=2, 
                 label=f'Epoch {epoch_num}', alpha=0.7)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Gradient Magnitude', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title('Evolution of Gradient Density Distribution during Training', fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # 保存圖表
    save_path = os.path.join(save_dir, 'gradient_density_evolution.png')
    plt.savefig(save_path)
    plt.close()
    print(f"梯度密度演變圖已保存至: {save_path}")

def visualize_training_history(history_file, save_dir='./ghm_visualizations'):
    """
    視覺化訓練歷史記錄
    
    參數:
        history_file: 訓練歷史檔案路徑
        save_dir: 保存視覺化結果的目錄
    """
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # 載入訓練歷史
        history = torch.load(history_file)
        
        # 繪製損失函數比較圖
        plt.figure(figsize=(14, 10))
        
        # 損失對比
        plt.subplot(2, 1, 1)
        plt.plot(history['epoch'], history['train_loss'], 'b-', linewidth=2, label='Train Regular Loss')
        plt.plot(history['epoch'], history['train_ghm_loss'], 'b--', linewidth=2, label='Train GHM Loss')
        plt.plot(history['epoch'], history['val_loss'], 'r-', linewidth=2, label='Val Regular Loss')
        plt.plot(history['epoch'], history['val_ghm_loss'], 'r--', linewidth=2, label='Val GHM Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=12)
        plt.title('Training and Validation Loss Comparison', fontsize=14)
        
        # 正則化損失比
        plt.subplot(2, 1, 2)
        train_ratio = [g/r if r > 0 else 1.0 for g, r in zip(history['train_ghm_loss'], history['train_loss'])]
        val_ratio = [g/r if r > 0 else 1.0 for g, r in zip(history['val_ghm_loss'], history['val_loss'])]
        
        plt.plot(history['epoch'], train_ratio, 'b-', linewidth=2, label='Train GHM/Regular Ratio')
        plt.plot(history['epoch'], val_ratio, 'r-', linewidth=2, label='Val GHM/Regular Ratio')
        plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('GHM/Regular Loss Ratio', fontsize=12)
        plt.legend(fontsize=12)
        plt.title('Relative Effect of GHM Reweighting', fontsize=14)
        
        plt.tight_layout()
        loss_path = os.path.join(save_dir, 'loss_comparison_detailed.png')
        plt.savefig(loss_path)
        plt.close()
        print(f"詳細損失比較圖已保存至: {loss_path}")
        
        # 繪製準確率趨勢圖
        plt.figure(figsize=(12, 6))
        plt.plot(history['epoch'], history['train_accuracy'], 'b-', linewidth=2, label='Train Accuracy')
        plt.plot(history['epoch'], history['val_accuracy'], 'r-', linewidth=2, label='Val Accuracy')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.legend(fontsize=12)
        plt.title('Training and Validation Accuracy', fontsize=14)
        
        # 添加數據標籤
        for i in range(0, len(history['epoch']), max(1, len(history['epoch'])//10)):
            plt.annotate(f"{history['train_accuracy'][i]:.1f}%", 
                        (history['epoch'][i], history['train_accuracy'][i]),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center')
            plt.annotate(f"{history['val_accuracy'][i]:.1f}%", 
                        (history['epoch'][i], history['val_accuracy'][i]),
                        textcoords="offset points", 
                        xytext=(0,-15), 
                        ha='center')
        
        plt.tight_layout()
        acc_path = os.path.join(save_dir, 'accuracy_trend.png')
        plt.savefig(acc_path)
        plt.close()
        print(f"準確率趨勢圖已保存至: {acc_path}")
        
    except Exception as e:
        print(f"載入或視覺化訓練歷史時出錯: {e}")

def compare_ghm_weights_distribution(plots_dir, save_dir='./ghm_visualizations'):
    """
    比較不同階段 GHM 權重分佈
    
    參數:
        plots_dir: 存放 GHM 繪圖結果的目錄
        save_dir: 保存視覺化結果的目錄
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 創建一個複合圖形以比較不同階段的梯度權重分佈
    fig = plt.figure(figsize=(16, 10))
    
    # 選取三個階段：早期、中期、晚期
    stages = {
        'Early': glob.glob(os.path.join(plots_dir, 'epoch[1-5]_batch0_*.png')),
        'Middle': glob.glob(os.path.join(plots_dir, 'epoch[2-3]0_batch0_*.png')),
        'Late': glob.glob(os.path.join(plots_dir, 'epoch[5-6]0_batch0_*.png'))
    }
    
    for i, (stage, files) in enumerate(stages.items()):
        if not files:
            continue
            
        # 選擇該階段的一個代表性文件
        sample_file = sorted(files)[-1]
        
        # 在組合圖中添加這個階段的圖像
        ax = fig.add_subplot(1, 3, i+1)
        img = plt.imread(sample_file)
        ax.imshow(img)
        ax.set_title(f"{stage} Stage", fontsize=14)
        ax.axis('off')
    
    plt.tight_layout()
    
    # 保存組合圖
    save_path = os.path.join(save_dir, 'ghm_weights_comparison.png')
    plt.savefig(save_path)
    plt.close()
    print(f"GHM 權重分佈比較圖已保存至: {save_path}")

def main():
    # 設置樣式
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # 設置頻率和材質
    frequency = '3000hz'
    material = 'plastic'
    
    # 設置目錄
    base_dir = os.path.join('saved_models')
    stats_dir = os.path.join(base_dir, 'ghm_stats', f"{material}_{frequency}")
    plots_dir = os.path.join(base_dir, 'ghm_plots', f"{material}_{frequency}")
    checkpoints_dir = os.path.join(base_dir, 'model_checkpoints', f"{material}_{frequency}_ghm")
    
    # 創建保存目錄
    save_dir = './ghm_visualizations'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"開始為 {frequency} 頻率的 {material} 材質生成 GHM 視覺化分析結果...")
    
    # 1. 視覺化梯度密度演變
    visualize_gradient_density_evolution(stats_dir, save_dir)
    
    # 2. 查找並視覺化訓練歷史
    history_files = glob.glob(os.path.join(checkpoints_dir, 'training_history_*.pt'))
    if history_files:
        visualize_training_history(history_files[0], save_dir)
    else:
        print(f"未找到訓練歷史檔案於 {checkpoints_dir}")
    
    # 3. 比較不同階段的 GHM 權重分佈
    compare_ghm_weights_distribution(plots_dir, save_dir)
    
    print(f"GHM 視覺化分析結果已保存至 {save_dir}")

if __name__ == "__main__":
    main() 