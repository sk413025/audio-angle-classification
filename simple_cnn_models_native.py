"""
簡單CNN模型定義：用於音頻頻譜圖分類
功能：
- 提供比ResNet更輕量級的CNN架構
- 包含頻譜圖適配器，將頻譜轉換為適合CNN處理的格式
- 維持與原始模型相同的接口，便於替換使用
- 提供分階段訓練控制（凍結/解凍網絡層）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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

class SimpleCNNAudioRanker(nn.Module):
    """
    簡單CNN架構的音頻排序模型 - 替代ResNet的輕量級選擇
    """
    def __init__(self, n_freqs=None):
        super().__init__()
        
        # 頻譜圖適配器 - 現在不再調整尺寸
        self.adapter = SpectrogramAdapter()
        
        # CNN特徵提取器 - 調整為處理原始尺寸的輸入
        self.features = nn.Sequential(
            # 第一層卷積區塊 - 使用較大的步長來減少特徵圖大小
            nn.Conv2d(1, 32, kernel_size=5, stride=(2, 4), padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二層卷積區塊
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三層卷積區塊
            nn.Conv2d(64, 128, kernel_size=3, stride=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第四層卷積區塊
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 自適應池化確保輸出大小一致
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 計算特徵大小
        self.feature_dim = 256
        
        # 分類器
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 輸出維度為1的排序分數
        )
        
        # 預設情況下，所有層都是可訓練的
        self.backbone_frozen = False

    def forward(self, x):
        # 通過適配器處理頻譜圖
        x = self.adapter(x)
        
        # 通過CNN提取特徵
        x = self.features(x)
        
        # 全局平均池化
        x = self.adaptive_pool(x)
        
        # 展平特徵
        x = x.view(x.size(0), -1)
        
        # 通過分類器產生排序分數
        x = self.classifier(x)
        
        return x
    
    def freeze_backbone(self):
        """凍結CNN骨幹網路層，只訓練分類器"""
        for param in self.features.parameters():
            param.requires_grad = False
            
        for param in self.classifier.parameters():
            param.requires_grad = True
            
        self.backbone_frozen = True
        print("CNN 骨幹已凍結，只訓練分類器層")
    
    def enable_full_finetune(self):
        """啟用所有層的微調"""
        for param in self.parameters():
            param.requires_grad = True
            
        self.backbone_frozen = False
        print("所有 CNN 層已解凍，進行全面微調")
    
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
