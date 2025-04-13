"""
快速測試 GHM 訓練腳本
"""

import os
import torch
from train_with_ghm import train_model_with_ghm
import config

def main():
    # 設置頻率和材質
    frequency = '1000hz'
    material = config.MATERIAL
    
    # 設置 GHM 參數
    ghm_params = {
        'bins': 10,
        'alpha': 0.75,
        'margin': config.MARGIN
    }
    
    # 設置訓練參數 - 只訓練少量輪數用於測試
    num_epochs = 3
    checkpoint_interval = 3
    
    print("快速測試 GHM 訓練...")
    print(f"使用頻率: {frequency}, 材質: {material}")
    print(f"GHM 參數: {ghm_params}")
    print(f"訓練輪數: {num_epochs}")
    
    # 訓練模型
    model, history = train_model_with_ghm(
        frequency, 
        material, 
        num_epochs=num_epochs, 
        checkpoint_interval=checkpoint_interval,
        ghm_params=ghm_params,
        verify_consistency=False
    )
    
    print("\n測試完成！")
    print(f"訓練損失: {history['train_loss'][-1]:.4f}, GHM 損失: {history['train_ghm_loss'][-1]:.4f}")
    print(f"驗證損失: {history['val_loss'][-1]:.4f}, GHM 損失: {history['val_ghm_loss'][-1]:.4f}")
    print(f"訓練準確率: {history['train_accuracy'][-1]:.2f}%, 驗證準確率: {history['val_accuracy'][-1]:.2f}%")

if __name__ == "__main__":
    main() 