import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GHMCLoss(nn.Module):
    """
    Gradient Harmonizing Mechanism for Classification Tasks
    
    Paper: https://arxiv.org/abs/1811.05181
    """
    def __init__(self, bins=10, alpha=0.75, epsilon=1e-7):
        super(GHMCLoss, self).__init__()
        self.bins = bins
        self.alpha = alpha
        self.epsilon = epsilon
        self.edges = torch.arange(0, bins + 1).float() / bins
        self.edges = self.edges.cuda() if torch.cuda.is_available() else self.edges
        self.weights = torch.zeros(bins).cuda() if torch.cuda.is_available() else torch.zeros(bins)
    
    def calc_gradient_weights(self, g):
        # Gradient norm for each sample
        g_norm = torch.abs(g)
        
        # Count samples in each bin
        n_samples = g.size(0)
        tot = torch.zeros(self.bins).to(g.device)
        
        # Compute statistics for bin assignment
        for i in range(self.bins):
            inds = (g_norm >= self.edges[i].to(g.device)) & (g_norm < self.edges[i+1].to(g.device))
            num_in_bin = inds.sum().item()
            tot[i] = num_in_bin
        
        # Compute weights
        tot = tot.clamp(min=1)
        w = tot.pow(-self.alpha)
        
        # Store weights for future use
        self.weights = w
        
        return w

    def forward(self, x, target):
        # Compute sigmoid outputs and binary cross entropy loss
        pred = torch.sigmoid(x)
        loss = F.binary_cross_entropy_with_logits(x, target, reduction='none')
        
        # Calculate gradients for GHM weighting
        g = torch.abs(pred - target)
        
        # Get weights based on gradient density
        w = self.calc_gradient_weights(g)
        
        # Assign weight to each sample based on its bin
        bin_idx = torch.clamp(torch.floor(g * self.bins).long(), 0, self.bins-1)
        sample_weights = w[bin_idx]
        
        # Apply weights to the loss
        weighted_loss = loss * sample_weights
        
        return weighted_loss.mean()

class GHMRLoss(nn.Module):
    """
    Gradient Harmonizing Mechanism for Regression Tasks
    
    Paper: https://arxiv.org/abs/1811.05181
    """
    def __init__(self, mu=0.02, bins=10, alpha=0.75, epsilon=1e-7):
        super(GHMRLoss, self).__init__()
        self.mu = mu
        self.bins = bins
        self.alpha = alpha
        self.epsilon = epsilon
        self.edges = torch.arange(0, bins + 1).float() / bins
        self.edges = self.edges.cuda() if torch.cuda.is_available() else self.edges
        self.weights = torch.zeros(bins).cuda() if torch.cuda.is_available() else torch.zeros(bins)
    
    def calc_gradient_weights(self, g):
        # Gradient norm for each sample
        g_norm = torch.abs(g)
        
        # Count samples in each bin
        n_samples = g.size(0)
        tot = torch.zeros(self.bins).to(g.device)
        
        # Compute statistics for bin assignment
        for i in range(self.bins):
            inds = (g_norm >= self.edges[i].to(g.device)) & (g_norm < self.edges[i+1].to(g.device))
            num_in_bin = inds.sum().item()
            tot[i] = num_in_bin
        
        # Compute weights
        tot = tot.clamp(min=1)
        w = tot.pow(-self.alpha)
        
        # Store weights for future use
        self.weights = w
        
        return w

    def forward(self, pred, target):
        # Compute smooth L1 loss
        diff = torch.abs(pred - target)
        loss = torch.where(diff < self.mu, 0.5 * diff * diff / self.mu, diff - 0.5 * self.mu)
        
        # Calculate gradients for GHM weighting
        g = torch.abs(torch.tanh(pred) - torch.tanh(target))
        
        # Get weights based on gradient density
        w = self.calc_gradient_weights(g)
        
        # Assign weight to each sample based on its bin
        bin_idx = torch.clamp(torch.floor(g * self.bins).long(), 0, self.bins-1)
        sample_weights = w[bin_idx]
        
        # Apply weights to the loss
        weighted_loss = loss * sample_weights
        
        return weighted_loss.mean()

class GHMRankingLoss(nn.Module):
    """
    Gradient Harmonizing Mechanism adapted for Ranking Tasks
    
    This class adapts GHM for ranking problems where we compare pairs of samples.
    """
    def __init__(self, margin=0.0, bins=10, alpha=0.75, epsilon=1e-7):
        super(GHMRankingLoss, self).__init__()
        self.margin = margin
        self.bins = bins
        self.alpha = alpha
        self.epsilon = epsilon
        self.edges = torch.arange(0, bins + 1).float() / bins
        self.weights = torch.zeros(bins)
        # Base ranking loss (without GHM)
        self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduction='none')
    
    def calc_gradient_weights(self, g):
        # Gradient norm for each sample
        g_norm = torch.abs(g)
        
        # Get device
        device = g.device
        
        # Move edges to the same device as g
        edges = self.edges.to(device)
        
        # Count samples in each bin
        n_samples = g.size(0)
        tot = torch.zeros(self.bins, device=device)
        
        # Compute statistics for bin assignment
        for i in range(self.bins):
            inds = (g_norm >= edges[i]) & (g_norm < edges[i+1])
            num_in_bin = inds.sum().item()
            tot[i] = num_in_bin
        
        # Compute weights
        tot = tot.clamp(min=1)
        w = tot.pow(-self.alpha)
        
        # Store weights for future use (on the same device as g)
        self.weights = w
        
        # 記錄每個 bin 的數量和權重，用於調試
        bin_counts = tot.cpu().numpy()
        bin_weights = w.cpu().numpy()
        
        # 返回計算的權重
        return w

    def forward(self, output1, output2, target):
        # 確保所有輸入在同一設備上
        device = output1.device
        output1 = output1.to(device)
        output2 = output2.to(device)
        target = target.to(device)
        
        # 計算標準 MarginRankingLoss
        loss = self.ranking_loss(output1, output2, target)
        
        # 計算樣本對的難度估計（作為梯度的替代）
        # 對於排序任務，我們使用預測的差異作為指標
        diff = output1 - output2
        
        # 調整符號基於目標
        # 對於 target=1，我們希望 output1 > output2
        # 對於 target=0 或 -1，我們希望 output1 < output2
        expected_sign = 2 * target - 1  # 將 0/1 轉換為 -1/1
        
        # 如果 (diff * expected_sign) 小，說明該樣本對較難區分
        # 使用梯度的近似估計
        g = torch.sigmoid(-diff * expected_sign + self.margin)
        
        # 基於梯度密度獲取權重
        w = self.calc_gradient_weights(g)
        
        # 根據梯度值為每個樣本分配到對應的 bin 和權重
        bin_idx = torch.clamp(torch.floor(g * self.bins).long(), 0, self.bins-1)
        sample_weights = w[bin_idx]
        
        # 應用權重到損失
        weighted_loss = loss * sample_weights
        
        return weighted_loss.mean() 