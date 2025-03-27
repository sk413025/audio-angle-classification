"""
ConvNeXt模型定義：用於音頻頻譜圖分類
功能：
- 提供基於ConvNeXt的現代CNN架構，替代簡單CNN
- 包含頻譜圖適配器，將頻譜轉換為適合ConvNeXt處理的格式
- 維持與原始模型相同的接口，便於替換使用
- 提供分階段訓練控制（凍結/解凍網絡層）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from functools import partial
from typing import Optional, Callable, List

class LayerNorm2d(nn.LayerNorm):
    """
    LayerNorm for channels of 2D spatial data
    """
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x

class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block as described in the paper
    https://arxiv.org/abs/2201.03545
    """
    def __init__(
        self,
        dim,
        layer_scale: float = 1e-6,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # Depthwise conv
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=9, dilation=3, groups=dim)
        
        # Inverted bottleneck
        self.norm = norm_layer(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # Pointwise/1x1 convs
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth_prob = stochastic_depth_prob

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.dwconv(x)
        
        # Channel mixing
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        
        x = self.layer_scale * x
        
        # Apply stochastic depth with probability
        if self.training and self.stochastic_depth_prob > 0.0:
            if torch.rand(1).item() < self.stochastic_depth_prob:
                return residual
            else:
                return residual + x
        else:
            return residual + x

class SpectrogramAdapter(nn.Module):
    """
    頻譜圖適配器 - 保持原始頻譜數據格式
    """
    def __init__(self, target_size=None):
        super().__init__()
        # 不再需要target_size，但保留參數以維持接口兼容性
        self.target_size = target_size
        
    def forward(self, x):
        # x 形狀: [batch, 1, freq, time] 例如 [N, 1, 513, 1126]
        
        # 只進行標準化，不調整尺寸
        x_mean = torch.mean(x)
        x_std = torch.std(x) + 1e-5
        x_norm = (x - x_mean) / x_std
        
        return x_norm

class ConvNeXtAudioRanker(nn.Module):
    """
    ConvNeXt架構的音頻排序模型 - 現代化的CNN選擇
    """
    def __init__(self, n_freqs=None):
        super().__init__()
        
        # 頻譜圖適配器
        self.adapter = SpectrogramAdapter()
        
        # 初始卷積層修改
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=(2, 2)),  # 降低時間維度的下採樣率
            LayerNorm2d(64),
        )
        
        # ConvNeXt主幹網路 - 恢復所有階段但調整深度
        self.stages = nn.Sequential(
            # Stage 1
            self._make_stage(64, 128, depth=2, stride=(2, 2)),
            
            # Stage 2
            self._make_stage(128, 256, depth=2, stride=(2, 2)),
            
            # Stage 3
            self._make_stage(256, 384, depth=4, stride=(2, 2)),
            
            # Stage 4
            self._make_stage(384, 512, depth=2, stride=(2, 2)),
        )
        
        # 自適應池化確保輸出大小一致
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 最終層正規化 - 修正為512通道
        self.norm = LayerNorm2d(512)
        
        # 特徵維度 - 修正為512
        self.feature_dim = 512
        
        # 分類器
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)  # 輸出維度為1的排序分數
        )
        
        # 初始化權重
        self.apply(self._init_weights)
        
        # 預設情況下，所有層都是可訓練的
        self.backbone_frozen = False

    def _make_stage(self, in_channels, out_channels, depth, stride):
        blocks = []
        
        # Downsampling layer
        downsample = nn.Sequential(
            LayerNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=stride, stride=stride),
        )
        blocks.append(downsample)
        
        # ConvNeXt blocks
        for i in range(depth):
            # 按深度增加隨機深度概率，添加更多正則化
            sd_prob = 0.0 if i == 0 else 0.1 * (i / depth)
            blocks.append(ConvNeXtBlock(
                out_channels, 
                layer_scale=1e-6, 
                stochastic_depth_prob=sd_prob
            ))
            
        return nn.Sequential(*blocks)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        # 通過適配器處理頻譜圖
        x = self.adapter(x)
        
        # Stem + ConvNeXt主幹網路
        x = self.stem(x)
        x = self.stages(x)
        
        # 正規化 + 池化
        x = self.norm(x)
        x = self.adaptive_pool(x)
        
        # 展平特徵
        x = x.flatten(1)
        
        # 通過分類器產生排序分數
        x = self.classifier(x)
        
        return x
    
    def freeze_backbone(self):
        """凍結ConvNeXt骨幹網路層，只訓練分類器"""
        for param in self.stem.parameters():
            param.requires_grad = False
            
        for param in self.stages.parameters():
            param.requires_grad = False
            
        for param in self.norm.parameters():
            param.requires_grad = False
            
        for param in self.classifier.parameters():
            param.requires_grad = True
            
        self.backbone_frozen = True
        print("ConvNeXt 骨幹已凍結，只訓練分類器層")
    
    def enable_full_finetune(self):
        """啟用所有層的微調"""
        for param in self.parameters():
            param.requires_grad = True
            
        self.backbone_frozen = False
        print("所有 ConvNeXt 層已解凍，進行全面微調")
    
    def print_trainable_parameters(self):
        """打印模型中哪些參數是可訓練的"""
        print("\n可訓練參數統計:")
        trainable_params = 0
        total_params = 0
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"{name}: 可訓練")
            else:
                print(f"{name}: 已凍結")
        
        print(f"可訓練參數總數: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")

# 為保持與原始模型的兼容性，創建別名
SimpleCNNAudioRanker = ConvNeXtAudioRanker 