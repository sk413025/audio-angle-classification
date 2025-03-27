import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非互動式後端
from matplotlib import pyplot as plt
import seaborn as sns
import sys

# 增加遞迴深度限制
sys.setrecursionlimit(10000)

def create_figure():
    """創建一個基本的圖形"""
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    return fig, ax

def plot_training_history(history_path):
    """
    繪製訓練歷史記錄的視覺化圖表
    
    參數:
        history_path: 訓練歷史記錄文件的路徑
    """
    # 加載訓練歷史記錄
    history = torch.load(history_path)
    
    # 將列表轉換為numpy數組
    epochs = np.array(history['epoch'])
    train_loss = np.array(history['train_loss'])
    val_loss = np.array(history['val_loss'])
    train_acc = np.array(history['train_accuracy'])
    val_acc = np.array(history['val_accuracy'])
    
    # 創建保存圖表的目錄
    plots_dir = os.path.join(os.path.dirname(history_path), 'training_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 從歷史記錄文件名中提取相關信息
    filename = os.path.basename(history_path)
    base_name = os.path.splitext(filename)[0]
    
    try:
        # 繪製損失曲線
        fig_loss, ax_loss = create_figure()
        ax_loss.plot(epochs, train_loss, 'b-', linewidth=2, marker='o', label='Training Loss')
        ax_loss.plot(epochs, val_loss, 'r-', linewidth=2, marker='s', label='Validation Loss')
        ax_loss.set_title('Model Loss Over Time', pad=20)
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.grid(True, linestyle='--', alpha=0.7)
        ax_loss.legend(loc='upper right')
        
        # 保存損失圖
        loss_plot_path = os.path.join(plots_dir, f"{base_name}_loss_plot.png")
        fig_loss.savefig(loss_plot_path, format='png', dpi=300)
        plt.close(fig_loss)
        
        # 繪製準確率曲線
        fig_acc, ax_acc = create_figure()
        ax_acc.plot(epochs, train_acc, 'b-', linewidth=2, marker='o', label='Training Accuracy')
        ax_acc.plot(epochs, val_acc, 'r-', linewidth=2, marker='s', label='Validation Accuracy')
        ax_acc.set_title('Model Accuracy Over Time', pad=20)
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy (%)')
        ax_acc.grid(True, linestyle='--', alpha=0.7)
        ax_acc.legend(loc='lower right')
        
        # 保存準確率圖
        acc_plot_path = os.path.join(plots_dir, f"{base_name}_accuracy_plot.png")
        fig_acc.savefig(acc_plot_path, format='png', dpi=300)
        plt.close(fig_acc)
        
        print(f"Training visualizations saved to:")
        print(f"Loss plot: {loss_plot_path}")
        print(f"Accuracy plot: {acc_plot_path}")
        
    except Exception as e:
        print(f"Error while plotting: {str(e)}")
        plt.close('all')  # 確保關閉所有圖形
        raise e

def print_history_content(history_path):
    """
    打印歷史記錄內容，用於調試
    """
    history = torch.load(history_path)
    print("\nHistory content:")
    for key, value in history.items():
        print(f"{key}: {type(value)}, shape/len: {len(value) if isinstance(value, list) else value.shape if hasattr(value, 'shape') else 'scalar'}")
        if len(value) > 0:
            print(f"First few values: {value[:5]}")

if __name__ == "__main__":
    # 示例用法
    import glob
    
    # 設定完整的路徑
    base_path = '/Users/sbplab/Hank/angle_classification_deg6/saved_models'
    
    # 獲取所有訓練歷史記錄文件
    history_files = glob.glob(os.path.join(base_path, 'training_history_*.pt'))
    
    if not history_files:
        print(f"No training history files found in {base_path}!")
        print("Current working directory:", os.getcwd())
        print("Files in directory:", os.listdir(base_path))
    else:
        print(f"Found {len(history_files)} history files:")
        for history_file in history_files:
            print(f"\nProcessing: {history_file}")
            try:
                # 首先打印歷史記錄內容以進行調試
                print_history_content(history_file)
                # 然後繪製圖表
                plot_training_history(history_file)
            except Exception as e:
                print(f"Error processing {history_file}: {str(e)}")
                import traceback
                traceback.print_exc() 