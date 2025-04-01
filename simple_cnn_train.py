"""
ResNet18模型訓練腳本
功能：
- 實現頻譜圖數據加載與預處理
- 配置並訓練ResNet18模型
- 保存訓練好的模型
- 評估模型性能
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime

from datasets import SpectrogramDatasetWithMaterial, RankingPairDataset
# from simple_cnn_models import ResNetAudioRanker
from simple_cnn_models_native import SimpleCNNAudioRanker as ResNetAudioRanker
# from convnext_models import ConvNeXtAudioRanker as ResNetAudioRanker

import config

def train_cnn_model(frequency, material, selected_seqs=config.SEQ_NUMS):
    """
    訓練ResNet18模型進行角度分類

    參數:
        frequency: 使用的頻率數據
        material: 使用的材質
        selected_seqs: 選擇的序列編號
    """
    print(f"開始訓練ResNet18模型 - 頻率: {frequency}, 材質: {material}")
    
    # 設置裝置
    device = config.DEVICE
    print(f"使用裝置: {device}")
    
    # 數據加載
    dataset = SpectrogramDatasetWithMaterial(
        os.path.join(config.DATA_ROOT, "step_018_sliced"),
        # ["deg000", "deg018", "deg036", "deg054", "deg072", "deg090", "deg108", "deg126", "deg144", "deg162", "deg180"],
        ["deg000", "deg036", "deg072", "deg108", "deg144", "deg180"],
        selected_seqs,
        frequency,
        material
    )
    
    if len(dataset) == 0:
        print("數據集為空，無法訓練模型")
        return
        
    # 添加訓練集和驗證集分割
    train_size = int(0.90 * len(dataset))  # 70% 作為訓練集
    val_size = len(dataset) - train_size
    
    # 確保訓練集和驗證集都至少有4個樣本（允許最小批次大小為2）
    if train_size < 4 or val_size < 4:
        print(f"數據集太小（總大小：{len(dataset)}），無法進行訓練/驗證分割")
        return
        
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 創建訓練集和驗證集的排序對數據集
    train_ranking_dataset = RankingPairDataset(train_dataset)
    # val_ranking_dataset = RankingPairDataset(val_dataset)
    val_ranking_dataset = RankingPairDataset(train_dataset)
    
    # 分別創建訓練和驗證數據加載器
    # 修改批次大小的計算方式
    min_dataset_size = min(len(train_ranking_dataset), len(val_ranking_dataset))
    batch_size = min(config.BATCH_SIZE, min_dataset_size)  # 使用較小的固定批次大小，或根據數據集大小調整
    
    print(f"訓練集排序對數量: {len(train_ranking_dataset)}")
    print(f"驗證集排序對數量: {len(val_ranking_dataset)}")
    print(f"選擇的批次大小: {batch_size}")
    
    train_dataloader = DataLoader(
        train_ranking_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        val_ranking_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True
    )
    
    # 檢查數據加載器是否為空
    if len(train_dataloader) == 0 or len(val_dataloader) == 0:
        print(f"數據加載器為空（訓練集大小：{len(train_ranking_dataset)}，驗證集大小：{len(val_ranking_dataset)}）")
        print(f"訓練數據加載器批次數: {len(train_dataloader)}")
        print(f"驗證數據加載器批次數: {len(val_dataloader)}")
        return

    print(f"數據集總大小: {len(dataset)}")
    print(f"訓練集大小: {len(train_dataset)}")
    print(f"驗證集大小: {len(val_dataset)}")
    print(f"訓練數據加載器批次數: {len(train_dataloader)}")
    print(f"驗證數據加載器批次數: {len(val_dataloader)}")
    print(f"使用的批次大小: {batch_size}")

    # 初始化模型
    model = ResNetAudioRanker(n_freqs=dataset.data.shape[2] if dataset.data is not None else None)
    model.to(device)
    
    # 打印模型信息
    print(model)
    model.print_trainable_parameters()
    
    # 損失函數和優化器
    criterion = nn.MarginRankingLoss(margin=config.MARGIN)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # 學習率調度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # 初始化記錄指標的列表
    training_history = {
        'epoch': [],
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    # 訓練循環
    best_loss = float('inf')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for epoch in range(config.NUM_EPOCHS):
        # 訓練階段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        print(f"\n===== Epoch {epoch+1}/{config.NUM_EPOCHS} =====")
        
        for i, (data1, data2, targets, label1, label2) in enumerate(train_dataloader):
            data1, data2 = data1.to(device), data2.to(device)
            targets = targets.to(device)
            
            # 清除梯度
            optimizer.zero_grad()
            
            # 前向傳播
            outputs1 = model(data1)
            outputs2 = model(data2)
            
            # 確保所有張量維度一致
            outputs1 = outputs1.view(-1)  # 展平為 [batch_size]
            outputs2 = outputs2.view(-1)  # 展平為 [batch_size]
            targets = targets.view(-1)    # 展平為 [batch_size]
            
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
        if len(train_dataloader) > 0:
            train_loss = train_loss / len(train_dataloader)
            train_accuracy = 100.0 * train_correct / train_total if train_total > 0 else 0.0
        else:
            train_loss = float('inf')
            train_accuracy = 0.0
        
        if len(val_dataloader) > 0:
            val_loss = val_loss / len(val_dataloader)
            val_accuracy = 100.0 * val_correct / val_total if val_total > 0 else 0.0
        else:
            val_loss = float('inf')
            val_accuracy = 0.0
        
        print(f"Epoch {epoch+1}")
        print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        
        # 使用驗證損失來更新學習率調度器
        # scheduler.step(val_loss)
        
        # 使用驗證損失來保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            model_path = os.path.join(
                config.SAVE_DIR,
                f"resnet18_{material}_{frequency}_best_{timestamp}.pt"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, model_path)
            print(f"Best model saved to {model_path}")
        
        # 每5個epoch記錄一次指標
        if (epoch + 1) % 5 == 0:
            training_history['epoch'].append(epoch + 1)
            training_history['train_loss'].append(train_loss)
            training_history['train_accuracy'].append(train_accuracy)
            training_history['val_loss'].append(val_loss)
            training_history['val_accuracy'].append(val_accuracy)
    
    # 保存訓練歷史記錄
    history_path = os.path.join(
        config.SAVE_DIR,
        f"training_history_{material}_{frequency}_{timestamp}.pt"
    )
    torch.save(training_history, history_path)
    print(f"Training history saved to {history_path}")
    
    return model, training_history

if __name__ == "__main__":
    # 訓練所有頻率下的模型
    for freq in config.FREQUENCIES:
        train_cnn_model(freq, config.MATERIAL)
