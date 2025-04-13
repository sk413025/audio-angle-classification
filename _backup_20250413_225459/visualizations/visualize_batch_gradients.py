"""
批次梯度分析與視覺化工具
功能：
- 載入不同訓練階段的模型檢查點
- 計算不同批次的梯度方向
- 分析批次梯度的一致性
- 視覺化梯度分布
- 計算批次梯度噪聲尺度(GNS)
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib
import glob
import re
matplotlib.use('Agg')  # Set backend

# Add the parent directory to the Python path
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import required modules
import config
from datasets import SpectrogramDatasetWithMaterial, RankingPairDataset
from simple_cnn_models import ResNetAudioRanker

def calculate_batch_gradients(model, criterion, batch_data1, batch_data2, batch_targets, device):
    """計算單個批次的梯度"""
    model.zero_grad()
    
    # 前向傳播
    outputs1 = model(batch_data1)
    outputs2 = model(batch_data2)
    
    # 確保所有張量維度一致
    outputs1 = outputs1.view(-1)
    outputs2 = outputs2.view(-1)
    batch_targets = batch_targets.view(-1)
    
    # 計算損失
    loss = criterion(outputs1, outputs2, batch_targets)
    
    # 反向傳播
    loss.backward()
    
    # 收集所有參數的梯度
    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            gradients.append(param.grad.view(-1).detach().cpu())
    
    # 將所有梯度連接成一個向量
    return torch.cat(gradients)

def calculate_gradient_noise_scale(model, criterion, dataloader, device, num_batches=None):
    """計算批次梯度噪聲尺度(GNS)"""
    model.train()
    all_gradients = []
    
    # 收集多個批次的梯度
    for i, (data1, data2, targets, _, _) in enumerate(dataloader):
        if num_batches is not None and i >= num_batches:
            break
            
        data1, data2 = data1.to(device), data2.to(device)
        targets = targets.to(device)
        
        grad = calculate_batch_gradients(model, criterion, data1, data2, targets, device)
        all_gradients.append(grad)
    
    # 計算平均梯度
    avg_gradient = torch.stack(all_gradients).mean(0)
    
    # 計算GNS
    gradient_variances = torch.stack([(grad - avg_gradient).pow(2) for grad in all_gradients])
    gradient_variance = gradient_variances.mean(0).mean()
    gradient_norm = avg_gradient.norm().pow(2)
    
    gns = gradient_variance / (gradient_norm + 1e-6)
    
    return gns.item(), all_gradients

def plot_gradient_directions(gradients, save_path):
    """使用t-SNE視覺化梯度方向"""
    # 將梯度轉換為numpy數組
    gradients_np = torch.stack(gradients).numpy()
    
    # 使用t-SNE降維，調整perplexity參數
    n_samples = len(gradients)
    perplexity = min(30, n_samples - 1)  # 確保perplexity小於樣本數
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    gradients_2d = tsne.fit_transform(gradients_np)
    
    # 繪製散點圖
    plt.figure(figsize=(10, 10))
    plt.scatter(gradients_2d[:, 0], gradients_2d[:, 1], alpha=0.6)
    plt.title(f'批次梯度方向的t-SNE視覺化\n(perplexity={perplexity})')
    plt.xlabel('t-SNE維度1')
    plt.ylabel('t-SNE維度2')
    
    # 保存圖片
    plt.savefig(save_path)
    plt.close()

def plot_gradient_cosine_similarities(gradients, save_path):
    """繪製梯度方向的余弦相似度矩陣"""
    n = len(gradients)
    similarities = torch.zeros((n, n))
    
    # 計算所有梯度對之間的余弦相似度
    for i in range(n):
        for j in range(n):
            cos_sim = torch.nn.functional.cosine_similarity(
                gradients[i].unsqueeze(0), 
                gradients[j].unsqueeze(0)
            )
            similarities[i, j] = cos_sim
    
    # 繪製熱力圖
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarities.numpy(), cmap='RdBu', center=0,
                xticklabels=False, yticklabels=False)
    plt.title('批次梯度方向的余弦相似度')
    plt.xlabel('批次索引')
    plt.ylabel('批次索引')
    
    # 保存圖片
    plt.savefig(save_path)
    plt.close()

def find_checkpoint_files(material, frequency):
    """查找指定材質和頻率的所有檢查點文件"""
    checkpoint_dir = os.path.join(config.SAVE_DIR, 'model_checkpoints', f"{material}_{frequency}")
    
    if not os.path.exists(checkpoint_dir):
        return []
    
    # 查找所有檢查點文件
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_ghm_epoch_*.pt"))
    
    # 提取epoch信息並排序
    def get_epoch(filename):
        match = re.search(r'model_epoch_(\d+)_', os.path.basename(filename))
        if match:
            return int(match.group(1))
        return 0
    
    checkpoint_files.sort(key=get_epoch)
    return checkpoint_files

def analyze_model_checkpoint(checkpoint_path, dataset, device, num_batches=None):
    """分析單個模型檢查點的梯度"""
    print(f"\n分析檢查點: {os.path.basename(checkpoint_path)}")
    
    # 加載檢查點
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 創建排序對數據集
    ranking_dataset = RankingPairDataset(dataset)
    print(f"排序對數量: {len(ranking_dataset)}")
    
    # 創建數據加載器
    batch_size = min(32, len(ranking_dataset))
    dataloader = DataLoader(
        ranking_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    # 初始化模型並加載權重
    # 獲取頻率維度 - 從原始數據集獲取
    n_freqs = dataset.data.shape[2]
    
    model = ResNetAudioRanker(n_freqs=n_freqs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # 損失函數
    criterion = torch.nn.MarginRankingLoss(margin=config.MARGIN)
    
    # 計算GNS和收集梯度
    total_batches = len(dataloader)
    batches_to_process = "全部" if num_batches is None else num_batches
    print(f"計算批次梯度噪聲尺度(GNS)... (處理 {batches_to_process}/{total_batches} 批次)")
    gns, gradients = calculate_gradient_noise_scale(model, criterion, dataloader, device, num_batches=num_batches)
    print(f"GNS = {gns:.6f}")
    
    # 獲取epoch號碼用於文件名
    epoch = checkpoint['epoch']
    
    # 生成視覺化
    print("生成視覺化圖表...")
    
    # 創建保存目錄
    material = config.MATERIAL
    frequency = dataset.selected_freq
    plots_dir = os.path.join(config.SAVE_DIR, 'batch_gradient_plots', f"{material}_{frequency}")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. 梯度方向的t-SNE視覺化
    tsne_path = os.path.join(plots_dir, f'gradient_directions_tsne_epoch_{epoch}.png')
    plot_gradient_directions(gradients, tsne_path)
    print(f"t-SNE視覺化已保存至: {tsne_path}")
    
    # 2. 梯度余弦相似度矩陣
    sim_path = os.path.join(plots_dir, f'gradient_similarities_epoch_{epoch}.png')
    plot_gradient_cosine_similarities(gradients, sim_path)
    print(f"相似度矩陣已保存至: {sim_path}")
    
    return gns

def main():
    print("開始批次梯度分析...")
    device = config.DEVICE
    print(f"使用設備: {device}")
    
    # 讓用戶選擇頻率
    available_frequencies = ['500hz', '1000hz', '3000hz']
    print("\n可用頻率:")
    for i, freq in enumerate(available_frequencies):
        print(f"{i+1}. {freq}")
    
    while True:
        try:
            choice = int(input("\n選擇頻率 (1-3): "))
            if 1 <= choice <= 3:
                selected_freq = available_frequencies[choice-1]
                break
            else:
                print("無效選擇，請輸入1-3之間的數字")
        except ValueError:
            print("請輸入有效數字")
    
    
    # 加載數據集
    dataset = SpectrogramDatasetWithMaterial(
        config.DATA_ROOT,
        config.CLASSES,
        config.SEQ_NUMS,
        selected_freq,
        config.MATERIAL
    )
    
    if len(dataset) == 0:
        print("數據集為空!")
        return
    
    print(f"數據集加載完成，共{len(dataset)}個樣本")

    train_size = int(0.70 * len(dataset))
    val_size = len(dataset) - train_size
    
    current_seed = config.SEED

    # Use a generator for reproducible splitting
    generator = torch.Generator().manual_seed(current_seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    dataset = train_dataset.dataset  # 使用訓練集作為驗證集，保持一致性
    
    
    # 查找檢查點文件
    checkpoints = find_checkpoint_files(config.MATERIAL, selected_freq)
    
    if not checkpoints:
        print(f"找不到任何 {config.MATERIAL}_{selected_freq} 的模型檢查點文件")
        print("請先運行 train_models_for_gradient_analysis.py 進行訓練")
        return
    
    print(f"\n找到 {len(checkpoints)} 個檢查點文件:")
    for i, cp in enumerate(checkpoints):
        print(f"{i+1}. {os.path.basename(cp)}")
    
    # 讓用戶選擇要分析的檢查點
    choice = input("\n選擇要分析的檢查點 (輸入編號，或輸入 'all' 分析所有檢查點): ")
    
    # 讓用戶選擇批次數量
    batch_choice = input("\n選擇要分析的批次數量 (輸入數字，或輸入 'all' 分析所有批次): ")
    num_batches = None
    if batch_choice.lower() != 'all':
        try:
            num_batches = int(batch_choice)
            print(f"將分析 {num_batches} 個批次")
        except ValueError:
            print("無效輸入，將分析所有批次")
    else:
        print("將分析所有批次")
    
    # 創建結果保存目錄
    results_dir = os.path.join(config.SAVE_DIR, 'batch_gradient_plots')
    os.makedirs(results_dir, exist_ok=True)
    
    # 分析選定的檢查點
    if choice.lower() == 'all':
        print("\n將分析所有檢查點")
        # 記錄每個檢查點的GNS
        epochs = []
        gns_values = []
        
        for cp in checkpoints:
            # 從文件名中提取epoch
            match = re.search(r'model_epoch_(\d+)_', os.path.basename(cp))
            epoch = int(match.group(1)) if match else 0
            epochs.append(epoch)
            
            # 分析檢查點
            gns = analyze_model_checkpoint(cp, dataset, device, num_batches=num_batches)
            gns_values.append(gns)
        
        # 繪製GNS隨epoch變化的曲線
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, gns_values, marker='o')
        title_suffix = "全部批次" if num_batches is None else f"前 {num_batches} 個批次"
        plt.title(f'批次梯度噪聲尺度(GNS)隨訓練進度變化 - {config.MATERIAL}_{selected_freq} ({title_suffix})')
        plt.xlabel('Epoch')
        plt.ylabel('GNS')
        plt.grid(True)
        
        # 保存圖表
        batch_info = "all_batches" if num_batches is None else f"first_{num_batches}_batches"
        gns_plot_path = os.path.join(results_dir, f'gns_vs_epoch_{config.MATERIAL}_{selected_freq}_{batch_info}.png')
        plt.savefig(gns_plot_path)
        plt.close()
        print(f"\nGNS變化曲線已保存至: {gns_plot_path}")
        
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(checkpoints):
                analyze_model_checkpoint(checkpoints[idx], dataset, device, num_batches=num_batches)
            else:
                print("無效選擇!")
        except ValueError:
            print("請輸入有效數字或 'all'")
    
    print("\n分析完成!")

if __name__ == "__main__":
    main() 