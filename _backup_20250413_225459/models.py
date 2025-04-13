"""
模型定義：包含所有神經網絡模型
功能：
- 將頻譜圖通過視覺提示方式轉換為ResNet可處理的格式
- 基於預訓練ResNet-18實現遷移學習
- 自適應處理不同大小的頻譜輸入
- 提供分階段訓練控制（凍結/解凍網絡層）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# FrequencyAttention class removed

class VisualPromptAdapter(nn.Module):
    def __init__(self, embed_size=(128, 128), position=(10, 10), init_scale=0.01):
        """
        視覺提示適配器 - 將頻譜數據嵌入到可學習的提示畫布中
        
        參數:
            embed_size: 嵌入頻譜的尺寸 (height, width)
            position: 嵌入頻譜在畫布中的位置 (y, x)
            init_scale: 可學習參數的初始化範圍
        """
        super().__init__()
        self.target_size = (224, 224)  # ResNet 需要的空間尺寸
        self.embed_size = embed_size
        self.position = position
        
        # 初始化可學習的提示畫布
        # 使用較小的初始化值以避免主導原始信號
        self.prompt_canvas = nn.Parameter(
            torch.randn(1, 3, self.target_size[0], self.target_size[1]) * init_scale
        )
        
    def forward(self, x):
        # x 形狀: [batch, 1, freq, time] 例如 [N, 1, 513, 1126]
        batch_size = x.size(0)
        
        # 1. 縮小頻譜到嵌入尺寸
        x_small = F.interpolate(x, size=self.embed_size, mode='bilinear', align_corners=False)
        
        # 2. 標準化頻譜數據
        x_mean = torch.mean(x_small)
        x_std = torch.std(x_small) + 1e-5
        x_norm = (x_small - x_mean) / x_std
        
        # 3. 為每個樣本創建個別的提示畫布副本
        prompt_batch = self.prompt_canvas.expand(batch_size, -1, -1, -1).clone()
        
        # 4. 將頻譜嵌入到畫布的指定位置
        y, x = self.position
        h, w = self.embed_size
        
        # 提取嵌入區域的邊界
        y_end = min(y + h, self.target_size[0])
        x_end = min(x + w, self.target_size[1])
        h_actual = y_end - y
        w_actual = x_end - x
        
        # 對所有三個通道嵌入相同的頻譜數據
        for c in range(3):
            prompt_batch[:, c, y:y_end, x:x_end] = x_norm[:, 0, :h_actual, :w_actual]
        
        return prompt_batch

class AudioRankerWithResNetPrompt(nn.Module):
    def __init__(self, n_freqs):
        super().__init__()
        # 視覺提示適配器 - 核心視覺提示組件
        self.adapter = VisualPromptAdapter(embed_size=(128, 128), position=(10, 10))
        
        # 載入預訓練的 ResNet-18
        self.resnet = models.resnet18(weights="IMAGENET1K_V1")
        
        # 替換最後一層為排序輸出
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # 輸出維度為1的排序分數
        )
        
        # 只凍結 ResNet 的骨幹網路層，保持 adapter 和 resnet.fc 可訓練
        self.freeze_backbone()

    def forward(self, x):
        # 通過視覺提示適配器轉換輸入
        x = self.adapter(x)
        
        # 通過 ResNet 提取特徵並生成排序分數
        x = self.resnet(x)
        
        return x
    
    def freeze_backbone(self):
        """凍結 ResNet 骨幹網路層，保持adapter 和 resnet.fc 可訓練"""
        # 凍結 ResNet 的層（除了 fc 層外）
        for name, module in self.resnet.named_children():
            if name != 'fc':  # 確保 fc 層保持可訓練
                for param in module.parameters():
                    param.requires_grad = False
        
        # 確保我們想要訓練的組件都是可訓練的
        for param in self.adapter.parameters():
            param.requires_grad = True
            
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
            
        print("ResNet 骨幹已凍結，只訓練視覺提示參數和分類器輸出層")
    
    def enable_full_finetune(self):
        """啟用所有層的微調"""
        for param in self.resnet.parameters():
            param.requires_grad = True
        print("所有 ResNet 層已解凍，進行全面微調")
    
    def print_trainable_parameters(self):
        """打印模型中哪些參數是可訓練的"""
        print("\n可訓練參數統計:")
        for name, param in self.named_parameters():
            print(f"{name}: {'可訓練' if param.requires_grad else '已凍結'}")
