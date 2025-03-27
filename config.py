"""
配置檔案：包含所有常量和設定
功能：
- 定義系統運行環境（MPS、CPU等）
- 設定音頻相關參數（採樣率）
- 指定數據路徑與分類角度
- 設定訓練參數（批次大小、訓練輪數）
- 設定模型保存位置
"""

import os
import torch

# 裝置設置
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 音頻相關常量
SAMPLE_RATE = 48000

# 數據路徑和分類
# 修改為適用於 macOS 的正確路徑格式
DATA_ROOT = "/Users/sbplab/Hank/20250211-方向性/step_036_sliced"  # 修改為正確的資料夾名稱
CLASSES = ["deg000", "deg036", "deg072", "deg108", "deg144", "deg180"]
# FREQUENCIES = ["500hz", "1000hz", "3000hz"]
FREQUENCIES = ["3000hz"]

SEQ_NUMS = ["00", "01", "02" ,"03" ,"04","05","06","07","08"]
# 新增材質類型
MATERIALS = ["box", "plastic"]  # 增加材質類型
MATERIAL = "plastic"  # 預設使用的材質

# 訓練相關常量
BATCH_SIZE = 4
NUM_EPOCHS = 20
LEARNING_RATE = 0.001  # 學習率
WEIGHT_DECAY = 1e-5  # 權重衰減（正則化）
MARGIN = 10.0  # 排名邊界值

# 模型保存位置
SAVE_DIR = "/Users/sbplab/Hank/angle_classification_deg6/saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

def print_system_info():
    """打印系統信息"""
    print(f"Using device: {DEVICE}")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
