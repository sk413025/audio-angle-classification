"""
模型訓練脚本 - 為梯度分析創建檢查點
功能：
- 訓練模型多個 epoch
- 每5個 epoch 儲存一次模型權重
- 支持不同頻率的模型訓練
- 記錄訓練過程中的損失和準確率
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

import config
from datasets import SpectrogramDatasetWithMaterial, RankingPairDataset
from simple_cnn_models import ResNetAudioRanker

def worker_init_fn(worker_id):
    """Function to make DataLoader workers deterministic."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Check for MPS availability and set seed
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    # Enable deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def verify_batch_consistency(dataloader, num_batches=5, save_path=None):
    """
    記錄dataloader的前幾個batch資料用於檢查一致性
    
    參數:
        dataloader: 資料加載器
        num_batches: 要記錄的batch數量
        save_path: 保存結果的檔案路徑
    
    返回:
        batch_data: 包含前num_batches個batch的資料
    """
    batch_data = []
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        # 只記錄batch的基本信息和識別特徵
        batch_info = {
            'batch_idx': i,
            'data1_shape': batch[0].shape,
            'data2_shape': batch[1].shape,
            'targets': batch[2].tolist(),
            'labels1': batch[3].tolist(),
            'labels2': batch[4].tolist(),
            # 添加資料的特徵值以檢查內容一致性
            'data1_sum': batch[0].sum().item(),
            'data2_sum': batch[1].sum().item(),
            'data1_std': batch[0].std().item(),
            'data2_std': batch[1].std().item()
        }
        batch_data.append(batch_info)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(batch_data, f)
        print(f"已保存batch資料到: {save_path}")
    
    return batch_data

def train_model_with_checkpoints(frequency, material, num_epochs=30, checkpoint_interval=5, seed=None, verify_consistency=False):
    """
    訓練模型並定期保存檢查點
    
    參數:
        frequency: 使用的頻率數據
        material: 使用的材質
        num_epochs: 訓練輪數，默認30
        checkpoint_interval: 每隔多少epoch保存一次模型，默認5
        seed: 用於可重複性的隨機種子
        verify_consistency: 是否驗證批次一致性
    """
    # Use provided seed or default from config
    current_seed = seed if seed is not None else config.SEED
    set_seed(current_seed)
    print(f"使用隨機種子: {current_seed}")

    print(f"開始訓練模型 - 頻率: {frequency}, 材質: {material}")
    
    # 設置裝置
    device = config.DEVICE
    print(f"使用裝置: {device}")
    
    # 數據加載
    dataset = SpectrogramDatasetWithMaterial(
        config.DATA_ROOT,
        config.CLASSES,
        config.SEQ_NUMS,
        frequency,
        material
    )
    
    if len(dataset) == 0:
        print("數據集為空，無法訓練模型")
        return
        
    # 添加訓練集和驗證集分割
    train_size = int(0.70 * len(dataset))
    val_size = len(dataset) - train_size
    
    # 確保訓練集和驗證集都至少有4個樣本
    if train_size < 4 or val_size < 4:
        print(f"數據集太小（總大小：{len(dataset)}），無法進行訓練/驗證分割")
        return
        
    # Use a generator for reproducible splitting
    generator = torch.Generator().manual_seed(current_seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    # 創建訓練集和驗證集的排序對數據集
    train_ranking_dataset = RankingPairDataset(train_dataset)
    val_ranking_dataset = RankingPairDataset(val_dataset)  # 使用訓練集作為驗證集，保持一致性
    
    # 分別創建訓練和驗證數據加載器
    min_dataset_size = min(len(train_ranking_dataset), len(val_ranking_dataset))
    batch_size = min(config.BATCH_SIZE, min_dataset_size)
    
    print(f"訓練集排序對數量: {len(train_ranking_dataset)}")
    print(f"驗證集排序對數量: {len(val_ranking_dataset)}")
    print(f"使用的批次大小: {batch_size}")
    
    train_dataloader = DataLoader(
        train_ranking_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(current_seed)
    )
    
    val_dataloader = DataLoader(
        val_ranking_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(current_seed)
    )
    
    # 檢查數據加載器是否為空
    if len(train_dataloader) == 0 or len(val_dataloader) == 0:
        print(f"數據加載器為空，無法訓練模型")
        return
    
    # 驗證批次一致性
    if verify_consistency:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        consistency_dir = os.path.join(config.SAVE_DIR, 'batch_consistency', f"{material}_{frequency}")
        os.makedirs(consistency_dir, exist_ok=True)
        
        # 記錄訓練批次資料
        train_batch_file = os.path.join(consistency_dir, f"train_batches_{timestamp}_seed{current_seed}.pkl")
        train_batch_data = verify_batch_consistency(train_dataloader, num_batches=3, save_path=train_batch_file)
        print(f"已記錄訓練批次資料，可通過檔案比較不同運行間的一致性")
    
    # 初始化模型
    model = ResNetAudioRanker(n_freqs=dataset.data.shape[2])
    model.to(device)
    
    # 損失函數和優化器
    criterion = nn.MarginRankingLoss(margin=config.MARGIN)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # 學習率調度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # 初始化記錄指標的列表
    training_history = {
        'epoch': [],
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'random_seed': current_seed
    }
    
    # 創建檢查點目錄
    checkpoint_dir = os.path.join(config.SAVE_DIR, 'model_checkpoints', f"{material}_{frequency}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 訓練循環
    for epoch in range(num_epochs):
        # 訓練階段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")
        
        for i, (data1, data2, targets, label1, label2) in enumerate(train_dataloader):
            data1, data2 = data1.to(device), data2.to(device)
            targets = targets.to(device)
            
            # 清除梯度
            optimizer.zero_grad()
            
            # 前向傳播
            outputs1 = model(data1)
            outputs2 = model(data2)
            
            # 確保所有張量維度一致
            outputs1 = outputs1.view(-1)
            outputs2 = outputs2.view(-1)
            targets = targets.view(-1)
            
            # 計算損失
            loss = criterion(outputs1, outputs2, targets)
            
            # 反向傳播
            loss.backward()
            
            # 更新參數
            optimizer.step()
            
            # 更新統計
            train_loss += loss.item()
            
            # 計算準確率
            predictions = (outputs1 > outputs2) == (targets > 0)
            train_correct += predictions.sum().item()
            train_total += targets.size(0)
            
            # 打印訓練進度
            if (i+1) % 10 == 0:
                print(f"Batch {i+1}/{len(train_dataloader)} | Loss: {loss.item():.4f}")
        
        # 驗證階段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data1, data2, targets, label1, label2 in val_dataloader:
                data1, data2 = data1.to(device), data2.to(device)
                targets = targets.to(device)
                
                outputs1 = model(data1)
                outputs2 = model(data2)
                
                # 確保所有張量維度一致
                outputs1 = outputs1.view(-1)
                outputs2 = outputs2.view(-1)
                targets = targets.view(-1)
                
                loss = criterion(outputs1, outputs2, targets)
                val_loss += loss.item()
                
                predictions = (outputs1 > outputs2) == (targets > 0)
                val_correct += predictions.sum().item()
                val_total += targets.size(0)
        
        # 計算並打印訓練和驗證指標
        train_loss = train_loss / len(train_dataloader) if len(train_dataloader) > 0 else float('inf')
        train_accuracy = 100.0 * train_correct / train_total if train_total > 0 else 0.0
        
        val_loss = val_loss / len(val_dataloader) if len(val_dataloader) > 0 else float('inf')
        val_accuracy = 100.0 * val_correct / val_total if val_total > 0 else 0.0
        
        print(f"Epoch {epoch+1}")
        print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        
        # 更新學習率調度器
        scheduler.step(val_loss)
        
        # 打印當前學習率
        for param_group in optimizer.param_groups:
            print(f"Current learning rate: {param_group['lr']:.6f}")
        
        # 保存訓練歷史
        training_history['epoch'].append(epoch + 1)
        training_history['train_loss'].append(train_loss)
        training_history['train_accuracy'].append(train_accuracy)
        training_history['val_loss'].append(val_loss)
        training_history['val_accuracy'].append(val_accuracy)
        
        # 每 checkpoint_interval 個epoch保存一次檢查點
        if (epoch + 1) % checkpoint_interval == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"model_epoch_{epoch+1}_{timestamp}.pt"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy
            }, checkpoint_path)
            print(f"模型檢查點已保存至: {checkpoint_path}")
    
    # 保存訓練歷史記錄
    history_path = os.path.join(
        checkpoint_dir,
        f"training_history_{timestamp}.pt"
    )
    torch.save(training_history, history_path)
    print(f"訓練歷史已保存至: {history_path}")
    
    return model, training_history

def check_batch_consistency():
    """檢查不同執行時batch是否一致"""
    frequency = '1000hz'  # 選擇一個頻率
    material = config.MATERIAL
    
    print(f"開始檢查batch一致性 - 頻率: {frequency}, 材質: {material}")
    
    # 使用相同種子執行兩次，比較批次是否一致
    print("第一次執行...")
    seed = 42
    train_model_with_checkpoints(frequency, material, num_epochs=1, checkpoint_interval=1, 
                                seed=seed, verify_consistency=True)
    
    print("\n第二次執行...")
    train_model_with_checkpoints(frequency, material, num_epochs=1, checkpoint_interval=1, 
                                seed=seed, verify_consistency=True)
    
    print("\n請比較生成的batch_consistency目錄下的檔案，檢查資料是否一致")
    print("如果兩次運行產生的批次資料完全相同，則表示批次順序與資料是固定的")

def main():
    # 可用頻率列表
    available_frequencies = ['500hz', '1000hz', '3000hz']
    
    # 讓用戶選擇頻率或檢查batch一致性
    print("\n選項:")
    print("1. 訓練模型")
    print("2. 檢查batch一致性")
    
    while True:
        try:
            option = int(input("\n選擇操作 (1 或 2): "))
            if 1 <= option <= 2:
                break
            else:
                print("無效選擇，請輸入1或2")
        except ValueError:
            print("請輸入有效數字")
    
    if option == 2:
        # 執行batch一致性檢查
        check_batch_consistency()
        return
    
    # 原始訓練流程
    print("\n可用頻率:")
    for i, freq in enumerate(available_frequencies):
        print(f"{i+1}. {freq}")
    
    while True:
        try:
            choice = int(input("\n選擇頻率 (1-3)，或輸入 0 訓練所有頻率: "))
            if 0 <= choice <= 3:
                break
            else:
                print("無效選擇，請輸入0-3之間的數字")
        except ValueError:
            print("請輸入有效數字")
    
    # 設置訓練參數
    num_epochs = int(input("\n設置訓練輪數 (推薦: 30): ") or "30")
    checkpoint_interval = int(input("設置檢查點保存間隔 (推薦: 5): ") or "5")
    verify = input("是否檢查批次一致性? (y/n): ").lower() == 'y'
    # Optionally ask for seed or use config default
    seed_to_use = config.SEED # Use default from config for simplicity here
    
    # 根據用戶選擇訓練模型
    if choice == 0:
        print("\n將依次訓練所有頻率的模型")
        for freq in available_frequencies:
            train_model_with_checkpoints(freq, config.MATERIAL, num_epochs, checkpoint_interval, 
                                        seed=seed_to_use, verify_consistency=verify)
    else:
        selected_freq = available_frequencies[choice-1]
        print(f"\n開始訓練頻率 {selected_freq} 的模型")
        train_model_with_checkpoints(selected_freq, config.MATERIAL, num_epochs, checkpoint_interval, 
                                    seed=seed_to_use, verify_consistency=verify)
    
    print("\n所有訓練完成！")

if __name__ == "__main__":
    # It's good practice to set a default seed if the script is run directly
    # This assumes config.py has a SEED defined, e.g., SEED = 42
    if not hasattr(config, 'SEED'):
        print("警告: config.py 中未定義 SEED。使用默認值 42。")
        config.SEED = 42
    main() 