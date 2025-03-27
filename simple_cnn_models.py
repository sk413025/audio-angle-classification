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
    頻譜圖適配器 - 將頻譜數據調整為適合CNN處理的格式
    """
    def __init__(self, target_size=(128, 128)):
        super().__init__()
        self.target_size = target_size
        
    def forward(self, x):
        # x 形狀: [batch, 1, freq, time] 例如 [N, 1, 513, 1126]
        
        # 調整頻譜圖尺寸
        x_resized = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        
        # 標準化頻譜數據
        x_mean = torch.mean(x_resized)
        x_std = torch.std(x_resized) + 1e-5
        x_norm = (x_resized - x_mean) / x_std
        
        return x_norm

class SimpleCNNAudioRanker(nn.Module):
    """
    簡單CNN架構的音頻排序模型 - 替代ResNet的輕量級選擇
    """
    def __init__(self, n_freqs=None):
        super().__init__()
        
        # 頻譜圖適配器
        self.adapter = SpectrogramAdapter(target_size=(128, 128))
        
        # CNN特徵提取器
        self.features = nn.Sequential(
            # 第一層卷積區塊
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二層卷積區塊
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三層卷積區塊
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
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
            nn.Dropout(0.5),
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
    頻譜圖適配器 - 將頻譜數據調整為適合CNN處理的格式
    """
    def __init__(self, target_size=(128, 128)):
        super().__init__()
        self.target_size = target_size
        
    def forward(self, x):
        # x 形狀: [batch, 1, freq, time] 例如 [N, 1, 513, 1126]
        
        # 調整頻譜圖尺寸
        x_resized = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        
        # 標準化頻譜數據
        x_mean = torch.mean(x_resized)
        x_std = torch.std(x_resized) + 1e-5
        x_norm = (x_resized - x_mean) / x_std
        
        return x_norm

class SimpleCNNAudioRanker(nn.Module):
    """
    簡單CNN架構的音頻排序模型 - 替代ResNet的輕量級選擇
    """
    def __init__(self, n_freqs=None):
        super().__init__()
        
        # 頻譜圖適配器
        self.adapter = SpectrogramAdapter(target_size=(128, 128))
        
        # CNN特徵提取器
        self.features = nn.Sequential(
            # 第一層卷積區塊
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二層卷積區塊
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三層卷積區塊
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
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
            nn.Dropout(0.5),
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
"""
簡單CNN模型定義：包含基於簡單卷積網絡的音頻排序模型
功能：
- 將頻譜圖通過視覺提示方式轉換為CNN可處理的格式
- 使用簡單的CNN架構代替ResNet進行特徵提取
- 自適應處理不同大小的頻譜輸入
- 提供分階段訓練控制（凍結/解凍網絡層）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.target_size = (224, 224)  # 保持與原模型相同的輸入尺寸
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

class AudioRankerWithSimpleCNN(nn.Module):
    def __init__(self, n_freqs=None):
        super().__init__()
        # 視覺提示適配器 - 與原模型相同
        self.adapter = VisualPromptAdapter(embed_size=(128, 128), position=(10, 10))
        
        # 簡單CNN架構替代ResNet
        self.features = nn.Sequential(
            # 第一個卷積塊
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # 第二個卷積塊
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # 第三個卷積塊
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 第四個卷積塊
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 全連接層輸出排序分數
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)  # 輸出維度為1的排序分數
        )
        
        # 爲了與原模型保持一致，提供backbone_layers屬性
        self.backbone_layers = [self.features]
        
    def forward(self, x):
        # 通過視覺提示適配器轉換輸入
        x = self.adapter(x)
        
        # 通過CNN提取特徵
        x = self.features(x)
        x = self.avgpool(x)
        
        # 通過全連接層生成排序分數
        x = self.classifier(x)
        
        return x
    
    def freeze_backbone(self):
        """凍結CNN骨幹網路層，保持adapter和classifier可訓練"""
        # 凍結特徵提取部分
        for param in self.features.parameters():
            param.requires_grad = False
        
        # 確保我們想要訓練的組件都是可訓練的
        for param in self.adapter.parameters():
            param.requires_grad = True
            
        for param in self.classifier.parameters():
            param.requires_grad = True
            
        print("CNN 骨幹已凍結，只訓練視覺提示參數和分類器輸出層")
    
    def enable_full_finetune(self):
        """啟用所有層的微調"""
        for param in self.parameters():
            param.requires_grad = True
        print("所有CNN層已解凍，進行全面微調")
    
    def print_trainable_parameters(self):
        """打印模型中哪些參數是可訓練的"""
        print("\n可訓練參數統計:")
        for name, param in self.named_parameters():
            print(f"{name}: {'可訓練' if param.requires_grad else '已凍結'}")
