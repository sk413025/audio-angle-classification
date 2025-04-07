"""
音頻頻譜圖分類模型：使用預訓練的ResNet18作為backbone
功能：
- 使用預訓練的ResNet18作為特徵提取器
- 包含頻譜圖適配器，將頻譜轉換為適合ResNet處理的格式
- 使用visual prompt進行提示嵌入
- 提供分階段訓練控制（凍結/解凍網絡層）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class SpectrogramAdapter(nn.Module):
    """
    頻譜圖適配器 - 將頻譜數據調整為適合ResNet處理的格式
    功能：
    - 調整頻譜圖尺寸
    - 使用多種正規化方法生成多通道表示
    - 將數據映射到符合ImageNet分佈的範圍
    """
    def __init__(self, target_size=(128, 128)):
        super().__init__()
        self.target_size = target_size
        
        # ImageNet正規化參數
        # self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        # self.imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
    def forward(self, x):
        # x 形狀: [batch, 1, freq, time] 例如 [N, 1, 513, 1126]
        
        # 1. 調整頻譜圖尺寸
        x_resized = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        
        # 2. 基本標準化 - 將數據縮放到[0,1]範圍
        x_min = torch.min(x_resized)
        x_max = torch.max(x_resized)
        x_range = x_max - x_min + 1e-5
        x_norm = (x_resized - x_min) / x_range  # 現在數據範圍是[0,1]
        
        # 3. 創建多通道表示，每個通道突出不同特徵
        # R通道: 原始標準化頻譜
        # G通道: 頻率增強版本（強調垂直方向）
        # B通道: 時間增強版本（強調水平方向）
        batch_size = x_norm.size(0)
        x_rgb = torch.zeros(batch_size, 3, self.target_size[0], self.target_size[1], 
                           device=x_norm.device)
        
        # R通道: 原始標準化頻譜
        x_rgb[:, 0:1, :, :] = x_norm
        
        # G通道: 頻率增強版本（使用對數變換來強調高頻段）
        x_log = torch.log(x_norm + 1e-5)
        x_log_norm = (x_log - torch.min(x_log)) / (torch.max(x_log) - torch.min(x_log) + 1e-5)
        x_rgb[:, 1:2, :, :] = x_log_norm
        
        # B通道: 時間增強版本（使用梯度來強調變化）
        # 計算水平方向的梯度（時間變化）
        x_grad = torch.zeros_like(x_norm)
        x_grad[:, :, :, 1:] = x_norm[:, :, :, 1:] - x_norm[:, :, :, :-1]
        x_grad_norm = (x_grad - torch.min(x_grad)) / (torch.max(x_grad) - torch.min(x_grad) + 1e-5)
        x_rgb[:, 2:3, :, :] = x_grad_norm
        
        # 4. 將數據調整為ImageNet的分佈範圍
        # 首先確保x_rgb設備與self.imagenet_mean設備相同
        device = x_rgb.device
        # imagenet_mean = self.imagenet_mean.to(device)
        # imagenet_std = self.imagenet_std.to(device)
        
        # 將[0,1]範圍映射到ImageNet分佈
        x_rgb = (x_rgb - 0.5) * 2  # 先映射到[-1,1]
        # x_rgb = x_rgb * imagenet_std + imagenet_mean  # 再映射到ImageNet分佈
        
        return x_rgb

class VisualPrompt(nn.Module):
    """
    視覺提示模板 - 提供可學習的提示畫布
    功能：
    - 初始化可學習的提示模板
    - 將輸入數據嵌入到模板的指定位置
    """
    def __init__(self, target_size=(224, 224), embed_size=(128, 128), position=(48, 48), init_scale=0.01):
        super().__init__()
        self.target_size = target_size
        self.embed_size = embed_size
        self.position_y, self.position_x = position  # 解包位置參數
        
        # 初始化可學習的prompt模板
        self.prompt_template = nn.Parameter(
            torch.randn(1, 3, self.target_size[0], self.target_size[1]) * init_scale
        )
    
    def forward(self, x, batch_size):
        """
        將輸入數據嵌入到prompt模板中
        
        參數:
            x: 輸入數據 [batch, channels, height, width]
            batch_size: 批次大小
        """
        # 為每個樣本創建個別的prompt模板副本
        prompt_batch = self.prompt_template.expand(batch_size, -1, -1, -1).clone()
        
        # 將數據嵌入到模板的指定位置
        pos_y, pos_x = self.position_y, self.position_x
        h, w = self.embed_size
        
        # 提取嵌入區域的邊界
        y_end = min(pos_y + h, self.target_size[0])
        x_end = min(pos_x + w, self.target_size[1])
        h_actual = y_end - pos_y
        w_actual = x_end - pos_x
        
        # 對所有三個通道嵌入相同的數據
        for c in range(3):
            prompt_batch[:, c, pos_y:y_end, pos_x:x_end] = x[:, c, :h_actual, :w_actual]
        
        return prompt_batch

class ResNetAudioRanker(nn.Module):
    """
    使用預訓練ResNet18的音頻排序模型
    """
    def __init__(self, n_freqs=None):
        super().__init__()
        
        # 頻譜圖適配器
        self.adapter = SpectrogramAdapter(target_size=(128, 128))
        
        # 視覺提示
        self.visual_prompt = VisualPrompt(
            target_size=(224, 224),
            embed_size=(128, 128),
            position=(48, 48),  # 將頻譜圖放在中心位置
            init_scale=0.01
        )
        
        # 載入預訓練的ResNet18
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # 移除原始分類器
        self.backbone.fc = nn.Identity()
        
        # 添加自定義分類器
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)  # 輸出維度為1的排序分數
        )
        
        # 預設情況下，凍結ResNet骨幹
        self.freeze_backbone()
        
    def forward(self, x):
        # 1. 通過適配器處理頻譜圖
        x = self.adapter(x)
        
        # 2. 使用visual prompt嵌入頻譜
        x = self.visual_prompt(x, x.size(0))
        
        # 3. 通過ResNet提取特徵
        x = self.backbone(x)
        
        # 4. 通過分類器產生排序分數
        x = self.classifier(x)
        
        return x
    
    def freeze_backbone(self):
        """凍結ResNet骨幹網路層，只訓練分類器和prompt模板"""
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        for param in self.classifier.parameters():
            param.requires_grad = True
            
        # 確保prompt模板可訓練
        for param in self.visual_prompt.parameters():
            param.requires_grad = True
            
        self.backbone_frozen = True
        print("ResNet 骨幹已凍結，只訓練分類器層和prompt模板")
    
    def enable_full_finetune(self):
        """啟用所有層的微調"""
        for param in self.parameters():
            param.requires_grad = True
            
        self.backbone_frozen = False
        print("所有層已解凍，進行全面微調")
    
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
