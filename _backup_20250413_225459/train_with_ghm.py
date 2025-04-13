"""
模型訓練脚本 - 使用 GHM (Gradient Harmonizing Mechanism) 為梯度分析創建檢查點
功能：
- 訓練模型多個 epoch
- 每5個 epoch 儲存一次模型權重
- 支持不同頻率的模型訓練
- 記錄訓練過程中的損失和準確率
- 使用 GHM 來優化梯度分佈

DEPRECATED: Core training logic moved to train.py. 
This file may contain outdated utility functions like check_batch_consistency.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys

import config
from datasets import SpectrogramDatasetWithMaterial, RankingPairDataset
from losses.ghm_loss import GHMRankingLoss
from utils import ghm_utils
from utils.common_utils import worker_init_fn, set_seed

def check_batch_consistency():
    """檢查不同執行時batch是否一致
    
    NOTE: This function currently calls train_model_with_ghm which is deprecated.
    It needs refactoring to work with train.py or other utilities.
    """
    frequency = '1000hz'  # 選擇一個頻率
    material = config.MATERIAL
    
    print(f"開始檢查batch一致性 - 頻率: {frequency}, 材質: {material}")
    
    # 使用相同種子執行兩次，比較批次是否一致
    print("第一次執行...")
    seed = 42
    # !!! THIS CALL WILL FAIL as train_model_with_ghm is removed !!!
    # Need to refactor how this check is performed.
    # Option 1: Call train.py as a subprocess with --verify-consistency
    # Option 2: Directly use utils.debugging_utils.verify_batch_consistency 
    #           after setting up a dataloader manually.
    # train_model_with_ghm(frequency, material, num_epochs=1, checkpoint_interval=1, 
    #                   seed=seed, verify_consistency=True)
    print("!!! check_batch_consistency needs refactoring to work with train.py !!!") 
    
    # print("\\n第二次執行...")
    # train_model_with_ghm(frequency, material, num_epochs=1, checkpoint_interval=1, 
    #                   seed=seed, verify_consistency=True)
    
    print("\n請比較生成的batch_consistency目錄下的檔案，檢查資料是否一致")
    print("如果兩次運行產生的批次資料完全相同，則表示批次順序與資料是固定的")

# Removed main function
# // ... main function definition removed ...

# Removed if __name__ == "__main__": block
# // ... if block removed ... 