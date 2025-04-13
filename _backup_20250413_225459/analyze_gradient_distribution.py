"""
批次梯度範數分佈分析工具
功能：
- 載入指定模型檢查點
- 使用檢查點中儲存的隨機種子確保數據載入的可重現性
- 計算每個批次的梯度向量的 L2 範數
- 視覺化梯度範數的分佈，特別關注長尾現象
- (可選) 識別梯度範數異常的批次
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg') # Set backend for environments without GUI

import argparse
import random
from scipy import stats

# Add the parent directory to the Python path (assuming this script is in a subdirectory)
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import required modules from the project
import config
from datasets import SpectrogramDatasetWithMaterial, RankingPairDataset
from simple_cnn_models import ResNetAudioRanker
import torch.nn as nn # Import nn for MarginRankingLoss

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
    # Deterministic settings (optional, can impact performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def calculate_batch_gradient_norm(model, criterion, batch_data1, batch_data2, batch_targets, device):
    """計算單個批次的梯度 L2 範數"""
    model.zero_grad() # Ensure gradients are cleared

    # Move data to device
    batch_data1, batch_data2 = batch_data1.to(device), batch_data2.to(device)
    batch_targets = batch_targets.to(device)

    # Forward pass
    outputs1 = model(batch_data1)
    outputs2 = model(batch_data2)

    # Ensure dimensions match for loss calculation
    outputs1 = outputs1.view(-1)
    outputs2 = outputs2.view(-1)
    batch_targets = batch_targets.view(-1)

    # Calculate loss
    loss = criterion(outputs1, outputs2, batch_targets)

    # Backward pass to compute gradients
    loss.backward()

    # Collect all gradients into a single vector and calculate norm
    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            gradients.append(param.grad.view(-1)) # No need to detach or move to cpu for norm calculation

    if not gradients:
        return 0.0 # Or handle as an error/warning

    full_grad_vector = torch.cat(gradients)
    grad_norm = torch.linalg.norm(full_grad_vector).item()

    # It's good practice to zero_grad again after calculation if the model might be used further
    model.zero_grad()

    return grad_norm

def plot_gradient_norm_distribution(gradient_norms, save_path, epoch):
    """Plot histogram and statistics of gradient norms"""
    if not gradient_norms:
        print("Warning: No gradient norms collected, cannot generate plot.")
        return

    # Calculate statistics
    mean_norm = np.mean(gradient_norms)
    median_norm = np.median(gradient_norms)
    std_norm = np.std(gradient_norms)
    max_norm = np.max(gradient_norms)
    min_norm = np.min(gradient_norms)
    p25_norm = np.percentile(gradient_norms, 25)
    p75_norm = np.percentile(gradient_norms, 75)
    p95_norm = np.percentile(gradient_norms, 95)
    p99_norm = np.percentile(gradient_norms, 99)
    skewness = stats.skew(gradient_norms)
    kurtosis = stats.kurtosis(gradient_norms)

    # Create 2x2 subplot layout
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Linear scale histogram
    ax1 = plt.subplot(2, 2, 1)
    sns.histplot(data=gradient_norms, kde=True, bins=50, ax=ax1)
    ax1.set_title('Gradient Norm Distribution (Linear Scale)')
    ax1.set_xlabel('Gradient L2 Norm')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    # 2. Log scale histogram (y-axis)
    ax2 = plt.subplot(2, 2, 2)
    sns.histplot(data=gradient_norms, kde=False, bins=50, log_scale=(False, True), ax=ax2)
    ax2.set_title('Gradient Norm Distribution (Log Y Scale)')
    ax2.set_xlabel('Gradient L2 Norm')
    ax2.set_ylabel('Frequency (Log Scale)')
    ax2.grid(True, which="both", ls="--", alpha=0.5)

    # 3. Log-log scale histogram
    ax3 = plt.subplot(2, 2, 3)
    sns.histplot(data=gradient_norms, kde=False, bins=50, log_scale=True, ax=ax3)
    ax3.set_title('Gradient Norm Distribution (Log-Log Scale)')
    ax3.set_xlabel('Gradient L2 Norm (Log Scale)')
    ax3.set_ylabel('Frequency (Log Scale)')
    ax3.grid(True, which="both", ls="--", alpha=0.5)

    # 4. Statistics text box
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    stats_text = f"""Gradient Norm Statistics (Epoch {epoch}):
    
Sample Size: {len(gradient_norms)}
    
Basic Statistics:
• Mean: {mean_norm:.2e}
• Median: {median_norm:.2e}
• Std Dev: {std_norm:.2e}
• Min: {min_norm:.2e}
• Max: {max_norm:.2e}

Percentiles:
• 25th: {p25_norm:.2e}
• 75th: {p75_norm:.2e}
• 95th: {p95_norm:.2e}
• 99th: {p99_norm:.2e}

Distribution Characteristics:
• Skewness: {skewness:.2f}
• Kurtosis: {kurtosis:.2f}
• IQR: {(p75_norm - p25_norm):.2e}
• CV: {(std_norm/mean_norm):.2f}
"""
    ax4.text(0.05, 0.95, stats_text, fontsize=12, va='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Gradient norm distribution plot saved to: {save_path}")
    print("\n" + stats_text)  # Also print statistics to console


def analyze_checkpoint(checkpoint_path, output_dir):
    """分析單個模型檢查點的梯度範數分佈"""
    print(f"分析檢查點: {checkpoint_path}")

    # Determine device
    device = config.DEVICE
    print(f"使用設備: {device}")

    # --- 1. Load Checkpoint --- 
    if not os.path.exists(checkpoint_path):
        print(f"錯誤: 檢查點文件不存在 {checkpoint_path}")
        return
    checkpoint = torch.load(checkpoint_path, map_location=device)
    epoch = checkpoint.get('epoch', '未知') # Get epoch for naming output files

    # --- MODIFIED: Read Seed from config.py instead of checkpoint ---
    if not hasattr(config, 'SEED'):
         print("錯誤: config.py 中未定義 SEED。無法確定要使用的隨機種子。")
         return
    analysis_seed = config.SEED
    print(f"警告: 直接從 config.py 讀取隨機種子: {analysis_seed}")
    print("       請確保此種子與生成此檢查點的訓練運行所使用的種子一致。")

    # --- 2. Initialize for Reproducibility ---
    set_seed(analysis_seed)
    print(f"已使用種子 {analysis_seed} 設置隨機狀態。")

    # --- 3. Load Data Consistently ---
    try:
        filename = os.path.basename(checkpoint_path)
        parent_dir_name = os.path.basename(os.path.dirname(checkpoint_path))
        material, frequency = parent_dir_name.split('_')
        print(f"從路徑推斷出 材質: {material}, 頻率: {frequency}")
    except Exception as e:
        print(f"錯誤: 無法從路徑 '{checkpoint_path}' 推斷材質和頻率: {e}")
        print("請確保檢查點保存在 'SAVE_DIR/model_checkpoints/MATERIAL_FREQUENCY/' 結構中")
        return

    # Initialize dataset
    dataset = SpectrogramDatasetWithMaterial(
        config.DATA_ROOT,
        config.CLASSES,
        config.SEQ_NUMS,
        frequency,
        material
    )

    if len(dataset) == 0:
        print("錯誤: 數據集為空!")
        return

    # Perform the exact same train/val split as in training
    train_size = int(0.90 * len(dataset))
    val_size = len(dataset) - train_size
    if train_size < 4 or val_size < 4:
        print(f"數據集太小 (總大小: {len(dataset)}), 無法執行與訓練時相同的分割。")
        return

    generator = torch.Generator().manual_seed(analysis_seed)
    train_dataset, _ = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    print(f"使用種子 {analysis_seed} 重新創建了訓練/驗證分割。訓練集大小: {len(train_dataset)}")

    # Create RankingPairDataset
    ranking_dataset = RankingPairDataset(train_dataset)
    print(f"創建排序對數據集，大小: {len(ranking_dataset)}")

    # Create DataLoader
    batch_size = min(config.BATCH_SIZE, len(ranking_dataset))
    if batch_size == 0:
        print("錯誤: 排序對數據集為空或 batch size 為 0。")
        return
    print(f"使用批次大小: {batch_size}")

    dataloader = DataLoader(
        ranking_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    print(f"數據加載器創建完成。批次數量: {len(dataloader)}")

    # --- 4. Load Model State ---
    try:
        n_freqs = dataset.data.shape[2]
    except AttributeError:
        print("警告: 無法自動獲取 n_freqs，將嘗試從第一個樣本獲取。")
        sample_data, _, _, _, _ = next(iter(dataloader))
        n_freqs = sample_data.shape[2]
        dataloader = DataLoader(
            ranking_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
        )

    model = ResNetAudioRanker(n_freqs=n_freqs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("模型架構初始化並加載權重完成。")

    # --- 5. Iterate and Calculate Gradient Norms ---
    criterion = nn.MarginRankingLoss(margin=config.MARGIN)
    batch_gradient_norms = []
    problematic_batches = []

    print(f"開始計算所有批次的梯度範數...")
    for i, (data1, data2, targets, label1, label2) in enumerate(dataloader):
        grad_norm = calculate_batch_gradient_norm(model, criterion, data1, data2, targets, device)
        batch_gradient_norms.append(grad_norm)

        # 識別異常高的梯度範數
        if i > 0 and grad_norm > 5 * np.mean(batch_gradient_norms[:-1]):
            problematic_batches.append({
                'batch_index': i,
                'norm': grad_norm,
                'labels': (label1, label2)
            })
            print(f"  批次 {i}: 檢測到高梯度範數 {grad_norm:.2e}")

        if (i + 1) % 20 == 0:
            print(f"  已處理批次 {i+1}/{len(dataloader)}...")

    print(f"梯度範數計算完成。共收集到 {len(batch_gradient_norms)} 個範數。")
    if problematic_batches:
        print("\n檢測到的異常批次:")
        for batch in problematic_batches:
            print(f"  批次 {batch['batch_index']}: 範數 = {batch['norm']:.2e}, 標籤對 = {batch['labels']}")

    # --- 6. Visualize Gradient Norm Distribution ---
    analysis_output_dir = os.path.join(output_dir, f"{material}_{frequency}", f"epoch_{epoch}_seed_{analysis_seed}")
    os.makedirs(analysis_output_dir, exist_ok=True)

    plot_filename = f"gradient_norm_distribution_epoch_{epoch}_seed_{analysis_seed}.png"
    plot_save_path = os.path.join(analysis_output_dir, plot_filename)

    plot_gradient_norm_distribution(batch_gradient_norms, plot_save_path, epoch)


def main():
    parser = argparse.ArgumentParser(description="分析模型檢查點的批次梯度範數分佈")
    parser.add_argument("checkpoint_path", type=str, help="要分析的模型檢查點文件路徑 (.pt)")
    parser.add_argument("--output_dir", type=str, default=os.path.join(config.SAVE_DIR, 'gradient_distribution_analysis'),
                        help="保存分析結果和圖表的目錄")
    args = parser.parse_args()

    # Ensure the base output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    analyze_checkpoint(args.checkpoint_path, args.output_dir)

if __name__ == "__main__":
    main() 