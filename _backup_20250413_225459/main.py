"""
主程式：執行訓練和評估
功能：
- 系統入口點，協調所有組件運行
- 資料集加載與檢查
- 模型初始化與配置
- 執行訓練與評估循環
- 結果視覺化與保存
- 錯誤處理與除錯支持
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import inspect

# 導入其他模組
from config import DEVICE, DATA_ROOT, CLASSES, BATCH_SIZE, NUM_EPOCHS, SAVE_DIR, LEARNING_RATE, WEIGHT_DECAY, MARGIN, print_system_info
from models import AudioRankerWithResNetPrompt
from datasets import RankingPairDataset, SpectrogramDatasetWithMaterial
from training import train_ranker, evaluate_ranker
from visualization import visualize_frequency_weights, plot_ranking_stats, save_ranking_results, visualize_attention_effect
# from model_utils import print_model_shapes  # 添加新导入

def main():
    # Add anomaly detection to help find gradient issues
    torch.autograd.set_detect_anomaly(True)
    
    # 打印系統信息
    print_system_info()
    
    # 明確從 config 導入需要的變數
    from config import CLASSES, SEQ_NUMS, DATA_ROOT, MATERIALS, MATERIAL
    
    selected_freq = '3000hz'
    selected_material = MATERIAL  # 使用預設材質或可以在這裡指定
    split_mode = 'custom'
    
    epochs = []
    train_losses = []
    test_losses = []
    train_ranking_accs = []
    test_ranking_accs = []
    
    # 確保保存目錄存在
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 加載數據集前進行路徑檢查
    print(f"Checking if data directory exists: {DATA_ROOT}")
    if os.path.exists(DATA_ROOT):
        print(f"Directory exists! Contents:")
        for root, dirs, files in os.walk(DATA_ROOT):
            level = root.replace(DATA_ROOT, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 4 * (level + 1)
            for f in files[:10]:  # 只顯示前10個文件，避免輸出過多
                print(f"{subindent}{f}")
            if len(files) > 10:
                print(f"{subindent}... ({len(files)-10} more files)")
    else:
        print(f"ERROR: Directory does not exist!")
        print("Current working directory:", os.getcwd())
        
    # 檢查 selected_freq 和 CLASSES 是否匹配目錄結構
    print(f"\nLooking for frequency: {selected_freq}")
    print(f"Looking for material: {selected_material}")
    print(f"Classes to find: {CLASSES}")
    
    # 嘗試找出一個預期的文件路徑作為示例
    example_path = os.path.join(DATA_ROOT, CLASSES[0], selected_material, 
                              f"{selected_material}_{CLASSES[0]}_{selected_freq}_{SEQ_NUMS[0]}.wav")
    print(f"Example expected file: {example_path}")
    print(f"This file exists: {os.path.exists(example_path)}")
    
    # 使用正確的參數名稱初始化數據集，加入材質參數
    print("\nLoading datasets...")
    
    # 檢查SpectrogramDatasetWithMaterial類的初始化參數
    sig = inspect.signature(SpectrogramDatasetWithMaterial.__init__)
    print(f"Dataset parameters: {list(sig.parameters.keys())[1:]}")  # 第一個是self，所以從1開始
    
    # 方法1: 按序列號分割
    train_seqs = SEQ_NUMS[:7]  # 使用序列號00-05進行訓練
    test_seqs = SEQ_NUMS[7:]   # 使用序列號06-08進行測試
    
    # 使用不同序列號加載訓練和測試數據
    print(f"Training sequences: {train_seqs}")
    print(f"Testing sequences: {test_seqs}")
    
    try:
        train_dataset = SpectrogramDatasetWithMaterial(
            data_dir=DATA_ROOT,
            classes=CLASSES,
            selected_seqs=train_seqs,  # 使用訓練序列
            selected_freq=selected_freq,
            material=selected_material
        )
        
        test_dataset = SpectrogramDatasetWithMaterial(
            data_dir=DATA_ROOT,
            classes=CLASSES,
            selected_seqs=test_seqs,  # 使用測試序列
            selected_freq=selected_freq,
            material=selected_material
        )
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        return
    
    # 在創建 DataLoader 之前檢查數據集大小
    print(f"\nTrain dataset raw size: {len(train_dataset)}")
    if len(train_dataset) == 0:
        print("WARNING: Train dataset is empty!")
        print("Make sure the data directory structure is correct:")
        print(f"Expected structure: {DATA_ROOT}/[class_name]/{selected_material}/{selected_freq}/[sequence_number].wav (or other extension)")
        return  # 如果數據集為空，終止程序
    
    # 顯示數據集中的一些關鍵屬性
    print("\nDataset details:")
    for attr in dir(train_dataset):
        if not attr.startswith('_') and not callable(getattr(train_dataset, attr)):
            try:
                value = getattr(train_dataset, attr)
                if not isinstance(value, (list, dict)) or len(str(value)) < 100:
                    print(f"  {attr}: {value}")
            except:
                pass
    
    # 創建配對數據集 - 使用新的RankingPairDataset代替PairDataset
    train_pair_dataset = RankingPairDataset(train_dataset)
    test_pair_dataset = RankingPairDataset(test_dataset)

    # 在載入數據集之後，添加以下代碼來檢查訓練和測試資料集的內容

    print("\n============ 訓練資料集前10筆檢查 ============")
    for i in range(min(50, len(train_pair_dataset))):
        data1, data2, target, label1, label2 = train_pair_dataset[i]
        print(data1[:, :5, :5])
        print(data2[:, :5, :5])
        print("-" * 50)

    # print("\n============ 測試資料集前10筆檢查 ============")
    # for i in range(min(50, len(test_pair_dataset))):
    #     data1, data2, target, label1, label2 = test_pair_dataset[i]
    #     print(data1[:, :5, :5])
    #     print(data2[:, :5, :5])
    #     print("-" * 50)

    # 增加檢查目標值的分佈情況
    train_targets = [train_pair_dataset[i][2].item() for i in range(len(train_pair_dataset))]
    test_targets = [test_pair_dataset[i][2].item() for i in range(len(test_pair_dataset))]

    print("\n============ 目標值分佈統計 ============")
    print(f"訓練資料集目標值 1 的比例: {train_targets.count(1) / len(train_targets):.2%}")
    print(f"訓練資料集目標值 -1 的比例: {train_targets.count(-1) / len(train_targets):.2%}")
    print(f"測試資料集目標值 1 的比例: {test_targets.count(1) / len(test_targets):.2%}")
    print(f"測試資料集目標值 -1 的比例: {test_targets.count(-1) / len(test_targets):.2%}")

    
    
    print(f"Train dataset raw size: {len(train_dataset)}")
    print(f"Train ranking pairs size: {len(train_pair_dataset)}")
    print(f"Test ranking pairs size: {len(test_pair_dataset)}")
    
    # 創建數據加載器
    train_loader = DataLoader(
        train_pair_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_pair_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    # 初始化模型
    print(f"Train dataset size: {len(train_pair_dataset)}")
    print(f"Test dataset size: {len(test_pair_dataset)}")
    
    # 初始化模型层的形状
    print("Initializing model...")
    model = AudioRankerWithResNetPrompt(len(CLASSES)).to(DEVICE)
    
    # 打印模型每层的形状
    # print_model_shapes(model, sample_data)
    
    # 定義損失函數和優化器
    criterion = nn.MarginRankingLoss(margin=MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 訓練和評估循環
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        # 訓練
        train_loss, train_ranking_acc = train_ranker(
            model, train_loader, optimizer, criterion, DEVICE, epoch
        )
        
        # 評估
        test_loss, test_ranking_acc = evaluate_ranker(
            model, test_loader, criterion, DEVICE
        )
        
        # 保存指標
        epochs.append(epoch + 1)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_ranking_accs.append(train_ranking_acc)
        test_ranking_accs.append(test_ranking_acc)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - "
              f"Train Loss: {train_loss:.4f}, Train Ranking Acc: {train_ranking_acc:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Ranking Acc: {test_ranking_acc:.4f}")
    
    # 保存模型
    model_save_path = os.path.join(SAVE_DIR, f"ranker_model_{selected_freq}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # 確保 model.freq_attention 有 freqs 屬性
    #if train_dataset.freqs is not None:
        # 獲取頻率值並設置到模型中
        #model.freq_attention.set_freqs(train_dataset.freqs)
        # 可視化頻率權重
        #visualize_frequency_weights(model, train_dataset.freqs, os.path.join(SAVE_DIR, f"freq_weights_{selected_freq}.png"))
    #else:
        #print("Warning: No frequency data available for visualization")
    
    # 可視化訓練結果 - 使用修改後的參數列表
    plot_ranking_stats(epochs, train_losses, test_losses, 
                      train_ranking_accs, test_ranking_accs, SAVE_DIR, selected_freq)
    
    # 儲存排名結果
    save_ranking_results(model, test_dataset, DEVICE, SAVE_DIR, selected_freq)
    
    
    print("Training and evaluation completed!")

if __name__ == "__main__":
    main()