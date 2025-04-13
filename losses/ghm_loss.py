import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GHMC_Loss(nn.Module):
    """
    Gradient Harmonized Multi-class Classification Loss.
    
    Args:
        bins (int): Number of bins to divide the gradient space.
        alpha (float): Parameter to control the focusing strength.
        momentum (float): Parameter for momentum update of the histogram.
    """
    def __init__(self, bins=10, alpha=0.75, momentum=0.9):
        super(GHMC_Loss, self).__init__()
        self.bins = bins
        self.alpha = alpha
        self.momentum = momentum
        self.edges = [float(x) / bins for x in range(bins + 1)]
        self.edges[-1] += 1e-6  # to include 1.0 within the last bin
        self.acc_sum = [0.0 for _ in range(bins)]
        
    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predicted logits, shape (N, C)
            target (torch.Tensor): Target classes, shape (N,)
        """
        # Convert target to one-hot encoding
        target_one_hot = F.one_hot(target, num_classes=pred.size(1)).float()
        
        # Apply sigmoid to get probabilities
        pred = F.softmax(pred, dim=1)
        
        # Calculate gradients
        g = torch.abs(pred - target_one_hot)
        
        # Update histogram
        total = g.shape[0] * g.shape[1]
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= self.edges[i]) & (g < self.edges[i + 1])
            num_in_bin = inds.sum().item()
            
            if num_in_bin > 0:
                self.acc_sum[i] = self.momentum * self.acc_sum[i] + (1 - self.momentum) * num_in_bin
                n += 1
        
        if n > 0:
            weights = torch.zeros_like(g)
            for i in range(self.bins):
                inds = (g >= self.edges[i]) & (g < self.edges[i + 1])
                if self.acc_sum[i] > 0:
                    weights[inds] = total / (n * self.acc_sum[i])
        else:
            weights = torch.ones_like(g)
        
        # Apply focusing parameter
        weights = weights ** self.alpha
        
        # Calculate loss
        loss = F.cross_entropy(pred, target, reduction='none')
        loss = loss * weights.sum(dim=1) / weights.shape[1]
        
        return loss.mean()

class GHMR_Loss(nn.Module):
    """
    Gradient Harmonized Regression Loss for angle regression.
    
    Args:
        bins (int): Number of bins to divide the gradient space.
        alpha (float): Parameter to control the focusing strength.
        momentum (float): Parameter for momentum update of the histogram.
        mu (float): Parameter to control the distance between angles.
    """
    def __init__(self, bins=10, alpha=0.75, momentum=0.9, mu=1.0):
        super(GHMR_Loss, self).__init__()
        self.bins = bins
        self.alpha = alpha
        self.momentum = momentum
        self.mu = mu
        self.edges = [float(x) / bins for x in range(bins + 1)]
        self.edges[-1] += 1e-6
        self.acc_sum = [0.0 for _ in range(bins)]
        
    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predicted angles, shape (N, 1)
            target (torch.Tensor): Target angles, shape (N, 1)
        """
        # Calculate absolute difference, accounting for circular nature of angles
        diff = torch.abs(pred - target)
        diff = torch.min(diff, 360.0 - diff)
        
        # Calculate gradients (normalized)
        g = diff / 180.0  # Normalize to [0, 1]
        
        # Update histogram
        total = g.numel()
        n = 0
        for i in range(self.bins):
            inds = (g >= self.edges[i]) & (g < self.edges[i + 1])
            num_in_bin = inds.sum().item()
            
            if num_in_bin > 0:
                self.acc_sum[i] = self.momentum * self.acc_sum[i] + (1 - self.momentum) * num_in_bin
                n += 1
        
        if n > 0:
            weights = torch.zeros_like(g)
            for i in range(self.bins):
                inds = (g >= self.edges[i]) & (g < self.edges[i + 1])
                if self.acc_sum[i] > 0:
                    weights[inds] = total / (n * self.acc_sum[i])
        else:
            weights = torch.ones_like(g)
        
        # Apply focusing parameter
        weights = weights ** self.alpha
        
        # Calculate loss using smooth L1 for robustness
        loss = F.smooth_l1_loss(pred, target, reduction='none', beta=self.mu)
        loss = loss * weights
        
        return loss.mean()

# Custom angle loss that accounts for the circular nature of angles
class CircularMSELoss(nn.Module):
    def __init__(self):
        super(CircularMSELoss, self).__init__()
        
    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predicted angles in degrees (0-359)
            target (torch.Tensor): Target angles in degrees (0-359)
        """
        # Convert to radians for easier calculation
        pred_rad = pred * np.pi / 180.0
        target_rad = target * np.pi / 180.0
        
        # Calculate sin and cos components
        pred_sin = torch.sin(pred_rad)
        pred_cos = torch.cos(pred_rad)
        target_sin = torch.sin(target_rad)
        target_cos = torch.cos(target_rad)
        
        # Mean squared error on the unit circle
        loss_sin = F.mse_loss(pred_sin, target_sin)
        loss_cos = F.mse_loss(pred_cos, target_cos)
        
        return loss_sin + loss_cos 