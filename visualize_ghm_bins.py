import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def visualize_ghm_bins(metadata_path='/Users/sbplab/Hank/sinetone_sliced/step_018_sliced/metadata/metadata_v1.0.json'):
    if not os.path.exists(metadata_path):
        print(f"錯誤: 找不到元數據文件 {metadata_path}")
        return

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # 收集所有epoch的bin統計信息
    epochs = defaultdict(lambda: defaultdict(int))
    max_epoch = 0
    max_bin = 0
    
    # 計算每個epoch的bin分配
    for sample_id, sample_data in metadata.items():
        if 'ghm_bins' in sample_data:
            bin_data = sample_data['ghm_bins']
            for epoch_str, bin_value in bin_data.items():
                # 跳過非數字的epoch
                if not epoch_str.isdigit():
                    continue
                epoch = int(epoch_str)
                max_epoch = max(max_epoch, epoch)
                max_bin = max(max_bin, bin_value)
                epochs[epoch][bin_value] += 1
    
    if not epochs:
        print("沒有找到GHM bin數據")
        return
    
    # 創建一個圖表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 繪製每個epoch的bin分佈熱圖
    bin_matrix = np.zeros((max_epoch, max_bin + 1))
    for epoch in range(1, max_epoch + 1):
        for bin_val in range(max_bin + 1):
            bin_matrix[epoch-1, bin_val] = epochs[epoch].get(bin_val, 0)
    
    # 計算每個bin的百分比
    bin_percents = np.zeros_like(bin_matrix)
    for i in range(bin_matrix.shape[0]):
        row_sum = bin_matrix[i].sum()
        if row_sum > 0:
            bin_percents[i] = (bin_matrix[i] / row_sum) * 100
    
    # 熱圖
    im = ax1.imshow(bin_percents, cmap='YlOrRd')
    ax1.set_title('GHM Bin分配隨Epoch變化的熱圖 (%)')
    ax1.set_xlabel('Bin值')
    ax1.set_ylabel('Epoch')
    ax1.set_xticks(range(max_bin + 1))
    ax1.set_yticks(range(max_epoch))
    ax1.set_yticklabels(range(1, max_epoch + 1))
    
    # 添加數值標籤
    for i in range(max_epoch):
        for j in range(max_bin + 1):
            if bin_percents[i, j] > 0:
                text = ax1.text(j, i, f"{bin_percents[i, j]:.1f}%",
                           ha="center", va="center", color="black" if bin_percents[i, j] < 50 else "white")
    
    plt.colorbar(im, ax=ax1, label='样本百分比 (%)')
    
    # 折線圖：顯示每個bin隨時間的變化
    for bin_val in range(max_bin + 1):
        bin_counts = [bin_percents[epoch-1, bin_val] for epoch in range(1, max_epoch + 1)]
        if any(bin_counts):  # 只繪製有數據的bin
            ax2.plot(range(1, max_epoch + 1), bin_counts, marker='o', label=f'Bin {bin_val}')
    
    ax2.set_title('每個Bin隨Epoch的變化趨勢')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('樣本百分比 (%)')
    ax2.set_xticks(range(1, max_epoch + 1))
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ghm_bin_visualization.png')
    print(f"視覺化已保存為 'ghm_bin_visualization.png'")
    plt.show()

if __name__ == "__main__":
    visualize_ghm_bins() 