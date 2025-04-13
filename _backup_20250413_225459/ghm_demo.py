"""
GHM 演示腳本：用於快速展示 GHM 的效果
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from ghm_loss import GHMRankingLoss

def generate_imbalanced_data(n_easy=200, n_hard=50, seed=42):
    """
    生成不平衡排序數據集
    
    參數:
        n_easy: 簡單樣本數量
        n_hard: 困難樣本數量
        seed: 隨機種子
    
    返回:
        data1, data2: 排序對
        targets: 目標值（1表示data1應優於data2，0表示相反）
        is_hard: 樣本是否為困難樣本
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 簡單樣本（差異明顯）
    x1_easy = torch.rand(n_easy) * 0.3 + 0.7  # 0.7-1.0
    x2_easy = torch.rand(n_easy) * 0.3        # 0.0-0.3
    targets_easy = torch.ones(n_easy)
    is_hard_easy = torch.zeros(n_easy)
    
    # 困難樣本（差異微小）
    x1_hard = torch.rand(n_hard) * 0.1 + 0.45  # 0.45-0.55
    x2_hard = torch.rand(n_hard) * 0.1 + 0.44  # 0.44-0.54
    targets_hard = torch.ones(n_hard)
    is_hard_hard = torch.ones(n_hard)
    
    # 組合數據
    data1 = torch.cat([x1_easy, x1_hard])
    data2 = torch.cat([x2_easy, x2_hard])
    targets = torch.cat([targets_easy, targets_hard])
    is_hard = torch.cat([is_hard_easy, is_hard_hard])
    
    # 隨機打亂數據
    indices = torch.randperm(n_easy + n_hard)
    data1 = data1[indices]
    data2 = data2[indices]
    targets = targets[indices]
    is_hard = is_hard[indices]
    
    return data1, data2, targets, is_hard

def train_and_compare(data1, data2, targets, is_hard, n_epochs=100, lr=0.01, save_dir='./ghm_demo_results'):
    """
    訓練模型並比較 GHM 與傳統損失函數
    
    參數:
        data1, data2: 排序對
        targets: 目標值
        is_hard: 樣本是否為困難樣本
        n_epochs: 訓練輪數
        lr: 學習率
        save_dir: 結果保存目錄
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 隨機分割訓練集和測試集
    n_samples = len(data1)
    n_train = int(0.8 * n_samples)
    
    # 確保訓練集和測試集都包含困難樣本
    train_indices = torch.randperm(n_samples)[:n_train]
    test_indices = torch.tensor([i for i in range(n_samples) if i not in train_indices])
    
    train_data1, train_data2 = data1[train_indices].to(device), data2[train_indices].to(device)
    train_targets = targets[train_indices].to(device)
    train_is_hard = is_hard[train_indices].to(device)
    
    test_data1, test_data2 = data1[test_indices].to(device), data2[test_indices].to(device)
    test_targets = targets[test_indices].to(device)
    test_is_hard = is_hard[test_indices].to(device)
    
    # 簡單線性模型
    model_reg = nn.Linear(1, 1).to(device)
    model_ghm = nn.Linear(1, 1).to(device)
    
    # 複製初始權重
    model_ghm.load_state_dict(model_reg.state_dict())
    
    # 損失函數
    criterion_reg = nn.MarginRankingLoss(margin=0.1)
    criterion_ghm = GHMRankingLoss(margin=0.1, bins=10, alpha=0.75)
    
    # 優化器
    optimizer_reg = optim.SGD(model_reg.parameters(), lr=lr)
    optimizer_ghm = optim.SGD(model_ghm.parameters(), lr=lr)
    
    # 記錄訓練過程
    history = {
        'epoch': [],
        'reg_loss': [],
        'ghm_loss': [],
        'reg_acc': [],
        'ghm_acc': [],
        'reg_hard_acc': [],
        'ghm_hard_acc': [],
        'reg_easy_acc': [],
        'ghm_easy_acc': []
    }
    
    # 訓練循環
    for epoch in range(n_epochs):
        # 訓練常規模型
        model_reg.train()
        optimizer_reg.zero_grad()
        
        train_data1_reshaped = train_data1.unsqueeze(1)
        train_data2_reshaped = train_data2.unsqueeze(1)
        
        outputs1_reg = model_reg(train_data1_reshaped).squeeze()
        outputs2_reg = model_reg(train_data2_reshaped).squeeze()
        
        loss_reg = criterion_reg(outputs1_reg, outputs2_reg, train_targets)
        loss_reg.backward()
        optimizer_reg.step()
        
        # 訓練 GHM 模型
        model_ghm.train()
        optimizer_ghm.zero_grad()
        
        outputs1_ghm = model_ghm(train_data1_reshaped).squeeze()
        outputs2_ghm = model_ghm(train_data2_reshaped).squeeze()
        
        # 修正：確保梯度計算正確
        loss_ghm = criterion_ghm(outputs1_ghm, outputs2_ghm, train_targets)
        loss_ghm.backward()
        optimizer_ghm.step()
        
        # 評估模型
        model_reg.eval()
        model_ghm.eval()
        
        with torch.no_grad():
            # 訓練集評估
            pred_reg = (outputs1_reg > outputs2_reg).float()
            pred_ghm = (outputs1_ghm > outputs2_ghm).float()
            
            acc_reg = (pred_reg == train_targets).float().mean().item()
            acc_ghm = (pred_ghm == train_targets).float().mean().item()
            
            # 分別計算困難樣本和簡單樣本的準確率
            hard_mask = train_is_hard.bool()
            easy_mask = ~hard_mask
            
            hard_acc_reg = (pred_reg[hard_mask] == train_targets[hard_mask]).float().mean().item() if hard_mask.sum() > 0 else 0
            hard_acc_ghm = (pred_ghm[hard_mask] == train_targets[hard_mask]).float().mean().item() if hard_mask.sum() > 0 else 0
            
            easy_acc_reg = (pred_reg[easy_mask] == train_targets[easy_mask]).float().mean().item() if easy_mask.sum() > 0 else 0
            easy_acc_ghm = (pred_ghm[easy_mask] == train_targets[easy_mask]).float().mean().item() if easy_mask.sum() > 0 else 0
        
        # 記錄結果
        history['epoch'].append(epoch + 1)
        history['reg_loss'].append(loss_reg.item())
        history['ghm_loss'].append(loss_ghm.item())
        history['reg_acc'].append(acc_reg * 100)
        history['ghm_acc'].append(acc_ghm * 100)
        history['reg_hard_acc'].append(hard_acc_reg * 100)
        history['ghm_hard_acc'].append(hard_acc_ghm * 100)
        history['reg_easy_acc'].append(easy_acc_reg * 100)
        history['ghm_easy_acc'].append(easy_acc_ghm * 100)
        
        # 打印進度
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}")
            print(f"Regular - Loss: {loss_reg.item():.4f}, Acc: {acc_reg*100:.2f}%, Hard Acc: {hard_acc_reg*100:.2f}%, Easy Acc: {easy_acc_reg*100:.2f}%")
            print(f"GHM - Loss: {loss_ghm.item():.4f}, Acc: {acc_ghm*100:.2f}%, Hard Acc: {hard_acc_ghm*100:.2f}%, Easy Acc: {easy_acc_ghm*100:.2f}%")
    
    # 測試集評估
    model_reg.eval()
    model_ghm.eval()
    
    with torch.no_grad():
        test_data1_reshaped = test_data1.unsqueeze(1)
        test_data2_reshaped = test_data2.unsqueeze(1)
        
        test_outputs1_reg = model_reg(test_data1_reshaped).squeeze()
        test_outputs2_reg = model_reg(test_data2_reshaped).squeeze()
        
        test_outputs1_ghm = model_ghm(test_data1_reshaped).squeeze()
        test_outputs2_ghm = model_ghm(test_data2_reshaped).squeeze()
        
        test_pred_reg = (test_outputs1_reg > test_outputs2_reg).float()
        test_pred_ghm = (test_outputs1_ghm > test_outputs2_ghm).float()
        
        test_acc_reg = (test_pred_reg == test_targets).float().mean().item()
        test_acc_ghm = (test_pred_ghm == test_targets).float().mean().item()
        
        # 分別計算困難樣本和簡單樣本的準確率
        test_hard_mask = test_is_hard.bool()
        test_easy_mask = ~test_hard_mask
        
        test_hard_acc_reg = (test_pred_reg[test_hard_mask] == test_targets[test_hard_mask]).float().mean().item() if test_hard_mask.sum() > 0 else 0
        test_hard_acc_ghm = (test_pred_ghm[test_hard_mask] == test_targets[test_hard_mask]).float().mean().item() if test_hard_mask.sum() > 0 else 0
        
        test_easy_acc_reg = (test_pred_reg[test_easy_mask] == test_targets[test_easy_mask]).float().mean().item() if test_easy_mask.sum() > 0 else 0
        test_easy_acc_ghm = (test_pred_ghm[test_easy_mask] == test_targets[test_easy_mask]).float().mean().item() if test_easy_mask.sum() > 0 else 0
    
    print("\n測試集結果:")
    print(f"Regular - Acc: {test_acc_reg*100:.2f}%, Hard Acc: {test_hard_acc_reg*100:.2f}%, Easy Acc: {test_easy_acc_reg*100:.2f}%")
    print(f"GHM - Acc: {test_acc_ghm*100:.2f}%, Hard Acc: {test_hard_acc_ghm*100:.2f}%, Easy Acc: {test_easy_acc_ghm*100:.2f}%")
    
    # 視覺化訓練過程
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # 1. 損失比較
    plt.figure(figsize=(14, 8))
    plt.plot(history['epoch'], history['reg_loss'], 'b-', linewidth=2, label='Regular Loss')
    plt.plot(history['epoch'], history['ghm_loss'], 'r-', linewidth=2, label='GHM Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.title('Loss Comparison: Regular vs GHM', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_comparison.png'))
    plt.close()
    
    # 2. 準確率比較 - 總體
    plt.figure(figsize=(14, 8))
    plt.plot(history['epoch'], history['reg_acc'], 'b-', linewidth=2, label='Regular Accuracy')
    plt.plot(history['epoch'], history['ghm_acc'], 'r-', linewidth=2, label='GHM Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=12)
    plt.title('Overall Accuracy Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'overall_accuracy.png'))
    plt.close()
    
    # 3. 困難樣本準確率比較
    plt.figure(figsize=(14, 8))
    plt.plot(history['epoch'], history['reg_hard_acc'], 'b-', linewidth=2, label='Regular - Hard Samples')
    plt.plot(history['epoch'], history['ghm_hard_acc'], 'r-', linewidth=2, label='GHM - Hard Samples')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=12)
    plt.title('Hard Samples Accuracy Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'hard_samples_accuracy.png'))
    plt.close()
    
    # 4. 簡單樣本和困難樣本的比較（兩種方法）
    plt.figure(figsize=(14, 8))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['epoch'], history['reg_easy_acc'], 'g-', linewidth=2, label='Easy Samples')
    plt.plot(history['epoch'], history['reg_hard_acc'], 'r-', linewidth=2, label='Hard Samples')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=12)
    plt.title('Regular Loss', fontsize=14)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['epoch'], history['ghm_easy_acc'], 'g-', linewidth=2, label='Easy Samples')
    plt.plot(history['epoch'], history['ghm_hard_acc'], 'r-', linewidth=2, label='Hard Samples')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=12)
    plt.title('GHM Loss', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'easy_vs_hard_comparison.png'))
    plt.close()
    
    # 5. 準確率比較摘要（測試集結果）
    labels = ['Overall', 'Easy Samples', 'Hard Samples']
    regular_results = [test_acc_reg*100, test_easy_acc_reg*100, test_hard_acc_reg*100]
    ghm_results = [test_acc_ghm*100, test_easy_acc_ghm*100, test_hard_acc_ghm*100]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, regular_results, width, label='Regular Loss', color='blue', alpha=0.7)
    plt.bar(x + width/2, ghm_results, width, label='GHM Loss', color='red', alpha=0.7)
    
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Test Set Accuracy Comparison', fontsize=14)
    plt.xticks(x, labels, fontsize=12)
    plt.legend(fontsize=12)
    
    # 添加數值標籤
    for i, v in enumerate(regular_results):
        plt.text(i - width/2, v + 1, f"{v:.1f}%", ha='center', fontsize=10)
    
    for i, v in enumerate(ghm_results):
        plt.text(i + width/2, v + 1, f"{v:.1f}%", ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'test_accuracy_summary.png'))
    plt.close()
    
    return history

def main():
    print("GHM 演示：生成不平衡數據並比較 GHM 與傳統損失函數")
    
    # 生成數據
    n_easy = 200  # 簡單樣本數量
    n_hard = 50   # 困難樣本數量
    
    data1, data2, targets, is_hard = generate_imbalanced_data(n_easy, n_hard, seed=42)
    
    # 檢查一下 GHM 損失函數
    print("\n檢查 GHM 損失函數...")
    ghm_loss = GHMRankingLoss(margin=0.1, bins=10, alpha=0.75)
    
    # 小樣本測試
    test_out1 = torch.tensor([0.9, 0.7, 0.2, 0.5, 0.8])
    test_out2 = torch.tensor([0.8, 0.6, 0.3, 0.6, 0.7])
    test_targets = torch.tensor([1.0, 1.0, -1.0, -1.0, 1.0])
    
    ghm_loss_val = ghm_loss(test_out1, test_out2, test_targets)
    print(f"GHM 測試損失值: {ghm_loss_val.item():.4f}")
    
    # 訓練並比較
    save_dir = './ghm_demo_results'
    history = train_and_compare(data1, data2, targets, is_hard, n_epochs=100, save_dir=save_dir)
    
    print(f"\nGHM 演示結果已保存至 {save_dir} 目錄")

if __name__ == "__main__":
    main() 