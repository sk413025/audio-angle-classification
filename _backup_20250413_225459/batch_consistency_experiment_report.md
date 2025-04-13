# 批次一致性檢查實驗報告

## 實驗目的

為了確保模型訓練中的可重現性，我們需要驗證在固定隨機種子的情況下，每次訓練過程中的數據批次(batch)順序和內容是否完全一致。這對於梯度分析等需要高度可重現性的研究至關重要。

## 實驗步驟

1. 修改訓練腳本以支援批次一致性檢查
2. 建立批次比較工具
3. 執行實驗並比較結果

## 一、修改訓練腳本

我們對原有的訓練腳本 `train_models_for_gradient_analysis.py` 進行了以下關鍵修改：

1. 啟用 PyTorch 的確定性行為：
   ```python
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   ```

2. 為 DataLoader 添加 worker 初始化函數，確保工作進程的隨機性也是確定的：
   ```python
   def worker_init_fn(worker_id):
       worker_seed = torch.initial_seed() % 2**32
       np.random.seed(worker_seed)
       random.seed(worker_seed)
   ```

3. 在 DataLoader 中指定固定的隨機生成器：
   ```python
   train_dataloader = DataLoader(
       train_ranking_dataset,
       batch_size=batch_size,
       shuffle=True,
       num_workers=4,
       drop_last=True,
       worker_init_fn=worker_init_fn,
       generator=torch.Generator().manual_seed(current_seed)
   )
   ```

4. 添加批次資料記錄功能：
   ```python
   def verify_batch_consistency(dataloader, num_batches=5, save_path=None):
       # 記錄前幾個批次的資料特徵
       batch_data = []
       for i, batch in enumerate(dataloader):
           if i >= num_batches:
               break
           # 記錄批次資訊和特徵值
           batch_info = {
               'batch_idx': i,
               'data1_shape': batch[0].shape,
               'data2_shape': batch[1].shape,
               'targets': batch[2].tolist(),
               'labels1': batch[3].tolist(),
               'labels2': batch[4].tolist(),
               'data1_sum': batch[0].sum().item(),
               'data2_sum': batch[1].sum().item(),
               'data1_std': batch[0].std().item(),
               'data2_std': batch[1].std().item()
           }
           batch_data.append(batch_info)
       
       # 如果提供保存路徑，將結果保存為 pickle 檔案
       if save_path:
           os.makedirs(os.path.dirname(save_path), exist_ok=True)
           with open(save_path, 'wb') as f:
               pickle.dump(batch_data, f)
       
       return batch_data
   ```

5. 添加批次一致性檢查功能：
   ```python
   def check_batch_consistency():
       # 使用相同種子執行兩次，比較批次是否一致
       frequency = '1000hz'
       material = config.MATERIAL
       seed = 42
       
       # 第一次執行
       train_model_with_checkpoints(frequency, material, num_epochs=1, 
                                   seed=seed, verify_consistency=True)
       
       # 第二次執行
       train_model_with_checkpoints(frequency, material, num_epochs=1, 
                                   seed=seed, verify_consistency=True)
   ```

## 二、建立批次比較工具

我們創建了一個獨立的 Python 腳本 `compare_batch_consistency.py` 用於比較兩個 pickle 檔案中記錄的批次資料：

```python
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
    """比較兩個批次資料是否相同"""
    differences = []
    
    # 檢查各項指標
    if batch1['batch_idx'] != batch2['batch_idx']:
        differences.append(f"批次索引不同")
        
    # ... 檢查各項特徵 ...
    
    return len(differences) == 0, differences


def compare_batch_files(file1, file2, tolerance=1e-6):
    """比較兩個批次檔案"""
    data1 = load_pickle_file(file1)
    data2 = load_pickle_file(file2)
    
    # ... 檢查批次數量和內容 ...
    
    return all_identical, total_differences


def find_latest_pkl_files(directory, n=2):
    """找到指定目錄下最新的n個pkl文件"""
    pkl_files = glob(os.path.join(directory, '*.pkl'))
    return sorted(pkl_files, key=os.path.getmtime, reverse=True)[:n]
```

此工具支援以下功能：
- 直接比較兩個指定的 pickle 檔案
- 自動找出目錄中最新的兩個 pickle 檔案並比較
- 設定浮點數比較的容忍度

## 三、實驗結果

我們對 plastic 材質、頻率為 1000Hz 的模型訓練進行了批次一致性檢查。在以下配置下：
- 固定隨機種子: 42
- 啟用確定性計算
- 控制 DataLoader 的隨機性

使用我們的比較工具檢查兩次訓練運行的批次資料：

```bash
python compare_batch_consistency.py --dir ./saved_models/batch_consistency/plastic_1000hz
```

結果顯示：

```
比較檔案:
1. ./saved_models/batch_consistency/plastic_1000hz/train_batches_20250411_182311_seed42.pkl
2. ./saved_models/batch_consistency/plastic_1000hz/train_batches_20250411_182249_seed42.pkl

結果: 批次資料完全相同！這表示每次訓練的批次順序和內容都是固定的。
```

這證實了在相同的隨機種子下，我們的訓練過程是完全可重現的。數據的分割、批次的順序和內容都完全一致，這確保了梯度分析的可靠性。

## 四、操作指南

要在您自己的環境中執行批次一致性檢查，請按照以下步驟操作：

1. **運行批次一致性檢查**
   ```bash
   python train_models_for_gradient_analysis.py
   # 選擇選項 2 進行批次一致性檢查
   ```

2. **比較生成的批次資料**
   ```bash
   python compare_batch_consistency.py --dir 批次檔案目錄
   ```
   例如：
   ```bash
   python compare_batch_consistency.py --dir ./saved_models/batch_consistency/steel_1000hz
   ```

3. **檢視結果**
   如果顯示批次資料完全相同，則證明訓練過程的可重現性得到保證。

## 五、建議

1. 始終使用固定的隨機種子進行實驗
2. 在訓練前確認批次一致性，尤其是在進行梯度分析等需要高度可重現性的實驗前
3. 如果批次不一致，請檢查：
   - 是否正確設置所有隨機種子
   - 是否啟用確定性計算
   - 資料集加載和處理是否存在非確定性行為

## 六、結論

本實驗證實了我們的訓練流程在固定隨機種子的情況下具有完全的可重現性，每次訓練的批次順序和內容都是一致的。這為後續的梯度分析等實驗提供了可靠的基礎。 