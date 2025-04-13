"""
測試 GHM (Gradient Harmonizing Mechanism) 的實現
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

from ghm_loss import GHMRankingLoss
import ghm_utils

def test_ghm_basic_functionality():
    """測試 GHM 的基本功能"""
    print("測試 GHM 基本功能...")
    
    # 創建 GHM ranking loss
    ghm_loss = GHMRankingLoss(margin=0.3, bins=10, alpha=0.75)
    
    # 創建一些測試數據
    outputs1 = torch.FloatTensor([0.9, 0.7, 0.1, 0.5, 0.8])
    outputs2 = torch.FloatTensor([0.8, 0.6, 0.2, 0.6, 0.7])
    
    # 目標: 如果 outputs1 > outputs2, 目標為 1, 否則為 0
    targets = torch.FloatTensor([1.0, 1.0, 0.0, 0.0, 1.0])
    
    # 計算 GHM loss
    loss = ghm_loss(outputs1, outputs2, targets)
    
    # 計算常規 MarginRankingLoss
    regular_loss = nn.MarginRankingLoss(margin=0.3)(outputs1, outputs2, targets)
    
    print(f"GHM Loss: {loss.item():.4f}")
    print(f"Regular Loss: {regular_loss.item():.4f}")
    
    # 檢查損失是否為標量
    assert loss.dim() == 0, "Loss should be a scalar"
    
    # 檢查 GHM 權重是否已經計算
    assert ghm_loss.weights.sum() > 0, "GHM weights should be calculated"
    
    # 可視化 GHM 權重分佈
    os.makedirs("./test_results", exist_ok=True)
    ghm_utils.plot_gradient_distribution(
        ghm_loss, save_dir="./test_results", name="test_ghm_weights"
    )
    
    print("基本功能測試完成!")
    return loss, regular_loss

def test_ghm_with_imbalanced_data():
    """測試 GHM 處理不平衡數據的能力"""
    print("\n測試 GHM 處理不平衡數據...")
    
    # 創建一個具有許多簡單樣本和少量困難樣本的數據集
    n_samples = 100
    
    # 設置種子以獲得可重複的結果
    torch.manual_seed(42)
    
    # 創建簡單樣本（差異很大）
    outputs1_easy = torch.rand(n_samples) + 0.5  # 值在 0.5 到 1.5 之間
    outputs2_easy = torch.rand(n_samples) * 0.3  # 值在 0 到 0.3 之間
    targets_easy = torch.ones(n_samples)  # 所有目標為 1（outputs1 > outputs2）
    
    # 創建困難樣本（差異很小）
    outputs1_hard = torch.rand(20) * 0.1 + 0.45  # 值在 0.45 到 0.55 之間
    outputs2_hard = torch.rand(20) * 0.1 + 0.44  # 值在 0.44 到 0.54 之間
    targets_hard = torch.ones(20)  # 所有目標為 1（outputs1 > outputs2）
    
    # 組合數據
    outputs1 = torch.cat([outputs1_easy, outputs1_hard])
    outputs2 = torch.cat([outputs2_easy, outputs2_hard])
    targets = torch.cat([targets_easy, targets_hard])
    
    # 創建 GHM loss 和常規 loss
    ghm_loss = GHMRankingLoss(margin=0.1, bins=20, alpha=0.75)
    regular_loss = nn.MarginRankingLoss(margin=0.1)
    
    # 計算損失
    loss_ghm = ghm_loss(outputs1, outputs2, targets)
    loss_regular = regular_loss(outputs1, outputs2, targets)
    
    print(f"不平衡數據 - GHM Loss: {loss_ghm.item():.4f}")
    print(f"不平衡數據 - Regular Loss: {loss_regular.item():.4f}")
    
    # 根據梯度大小計算樣本權重
    diff = outputs1 - outputs2
    expected_sign = 2 * targets - 1
    g = torch.abs(torch.tanh(diff) - torch.tanh(ghm_loss.margin * expected_sign))
    
    # 可視化梯度分佈
    plt.figure(figsize=(10, 6))
    plt.hist(g.detach().numpy(), bins=30, alpha=0.7)
    plt.axvline(x=0.5, color='r', linestyle='--')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Gradient Magnitude')
    plt.ylabel('Count')
    plt.title('Gradient Distribution in Imbalanced Dataset')
    plt.savefig("./test_results/imbalanced_gradient_distribution.png")
    
    # 獲取 GHM 權重
    ghm_utils.plot_gradient_distribution(
        ghm_loss, save_dir="./test_results", name="imbalanced_ghm_weights"
    )
    
    print("不平衡數據測試完成!")
    return loss_ghm, loss_regular

def test_ghm_training_effect():
    """測試 GHM 對訓練過程的影響"""
    print("\n測試 GHM 對訓練過程的影響...")
    
    # 簡單的排序任務數據
    n_samples = 50
    n_features = 10
    n_epochs = 20
    
    # 設置種子以獲得可重複的結果
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 創建一個簡單的數據集
    # 特徵向量和權重向量
    X = torch.randn(n_samples, n_features)
    true_weights = torch.FloatTensor([0.1, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, -0.3, -0.4, -0.5])
    
    # 計算得分
    scores = X @ true_weights
    
    # 創建排序對
    indices = torch.randperm(n_samples)
    X1 = X[indices[:n_samples//2]]
    X2 = X[indices[n_samples//2:n_samples]]
    scores1 = scores[indices[:n_samples//2]]
    scores2 = scores[indices[n_samples//2:n_samples]]
    targets = (scores1 > scores2).float()
    
    # 定義一個簡單的線性模型
    class LinearModel(nn.Module):
        def __init__(self):
            super(LinearModel, self).__init__()
            self.linear = nn.Linear(n_features, 1)
            # 初始化權重為接近零的小值
            nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.linear.bias)
            
        def forward(self, x):
            return self.linear(x).squeeze()
    
    # 創建兩個相同的模型
    model_regular = LinearModel()
    model_ghm = LinearModel()
    
    # 複製初始權重以確保兩個模型從相同的起點開始
    model_ghm.load_state_dict(model_regular.state_dict())
    
    # 創建兩種損失函數
    regular_loss = nn.MarginRankingLoss(margin=0.1)
    ghm_loss = GHMRankingLoss(margin=0.1, bins=10, alpha=0.75)
    
    # 創建優化器
    optimizer_regular = torch.optim.SGD(model_regular.parameters(), lr=0.01)
    optimizer_ghm = torch.optim.SGD(model_ghm.parameters(), lr=0.01)
    
    # 記錄訓練過程
    history = {
        'regular_loss': [],
        'ghm_loss': [],
        'regular_accuracy': [],
        'ghm_accuracy': []
    }
    
    # 訓練模型
    for epoch in range(n_epochs):
        # 訓練常規模型
        model_regular.train()
        optimizer_regular.zero_grad()
        outputs1 = model_regular(X1)
        outputs2 = model_regular(X2)
        loss_regular = regular_loss(outputs1, outputs2, targets)
        loss_regular.backward()
        optimizer_regular.step()
        
        # 訓練 GHM 模型
        model_ghm.train()
        optimizer_ghm.zero_grad()
        outputs1_ghm = model_ghm(X1)
        outputs2_ghm = model_ghm(X2)
        loss_ghm = ghm_loss(outputs1_ghm, outputs2_ghm, targets)
        loss_ghm.backward()
        optimizer_ghm.step()
        
        # 計算準確率
        predictions_regular = (outputs1 > outputs2) == (targets > 0)
        accuracy_regular = predictions_regular.float().mean().item() * 100
        
        predictions_ghm = (outputs1_ghm > outputs2_ghm) == (targets > 0)
        accuracy_ghm = predictions_ghm.float().mean().item() * 100
        
        # 記錄結果
        history['regular_loss'].append(loss_regular.item())
        history['ghm_loss'].append(loss_ghm.item())
        history['regular_accuracy'].append(accuracy_regular)
        history['ghm_accuracy'].append(accuracy_ghm)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}")
            print(f"Regular - Loss: {loss_regular.item():.4f}, Accuracy: {accuracy_regular:.2f}%")
            print(f"GHM - Loss: {loss_ghm.item():.4f}, Accuracy: {accuracy_ghm:.2f}%")
    
    # 繪製損失和準確率的比較圖
    plt.figure(figsize=(12, 10))
    
    # 損失比較
    plt.subplot(2, 1, 1)
    plt.plot(range(1, n_epochs+1), history['regular_loss'], 'b-', label='Regular Loss')
    plt.plot(range(1, n_epochs+1), history['ghm_loss'], 'r-', label='GHM Loss')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Comparison')
    
    # 準確率比較
    plt.subplot(2, 1, 2)
    plt.plot(range(1, n_epochs+1), history['regular_accuracy'], 'b-', label='Regular Accuracy')
    plt.plot(range(1, n_epochs+1), history['ghm_accuracy'], 'r-', label='GHM Accuracy')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Comparison')
    
    plt.tight_layout()
    plt.savefig("./test_results/training_comparison.png")
    
    print("\n訓練效果測試完成!")
    print(f"最終準確率 - Regular: {history['regular_accuracy'][-1]:.2f}%, GHM: {history['ghm_accuracy'][-1]:.2f}%")
    
    return history

if __name__ == "__main__":
    print("開始測試 GHM 損失函數...")
    
    # 測試基本功能
    test_ghm_basic_functionality()
    
    # 測試不平衡數據
    test_ghm_with_imbalanced_data()
    
    # 測試對訓練的影響
    history = test_ghm_training_effect()
    
    print("\n所有測試完成!")
    print("請檢查 './test_results' 目錄中的圖片以查看更多詳細信息。") 