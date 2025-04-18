"""
实验配置模块
用于集中管理SVRG优化器比较实验的参数
"""

import os
import torch
from datetime import datetime

# 实验基本配置
EXPERIMENT_NAME = "svrg_comparison"
BASE_SAVE_DIR = os.path.join("experiments", "svrg_comparison", "results")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EXPERIMENT_DIR = os.path.join(BASE_SAVE_DIR, f"{EXPERIMENT_NAME}_{TIMESTAMP}")

# 训练配置
EPOCHS = 30
CHECKPOINT_INTERVAL = 5  # 每隔多少个epoch保存检查点
BATCH_SIZE = 32
SEED = 42

# 优化器配置
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
SGD_MOMENTUM = 0.9
SGD_NESTEROV = False

# 数据集配置
FREQUENCIES = ["500hz", "1000hz", "3000hz"]
TRAIN_RATIO = 0.7  # 训练集比例
VAL_RATIO = 0.3    # 验证集比例

# 损失函数配置
MARGIN = 1.0  # 排序损失的边界值

# 日志配置
LOG_INTERVAL = 10  # 每隔多少个batch输出日志

# 异常样本阈值
HIGH_LOSS_PERCENTILE = 0.9  # 损失值位于前10%的样本被标记为高损失样本

# 设备配置
def get_device():
    """获取可用的计算设备"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

DEVICE = get_device()

# 创建实验目录
def create_experiment_dirs():
    """创建实验所需的目录结构"""
    # 主实验目录
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    
    # 为每种优化器和频率创建子目录
    for optimizer_type in ["svrg", "standard", "sgd_momentum"]:
        for frequency in FREQUENCIES:
            # 检查点目录
            checkpoint_dir = os.path.join(EXPERIMENT_DIR, optimizer_type, frequency, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # 批次记录目录
            batch_dir = os.path.join(EXPERIMENT_DIR, optimizer_type, frequency, "batch_records")
            os.makedirs(batch_dir, exist_ok=True)
            
            # 样本记录目录
            sample_dir = os.path.join(EXPERIMENT_DIR, optimizer_type, frequency, "sample_records")
            os.makedirs(sample_dir, exist_ok=True)
            
            # 指标目录
            metrics_dir = os.path.join(EXPERIMENT_DIR, optimizer_type, frequency, "metrics")
            os.makedirs(metrics_dir, exist_ok=True)
            
            # 梯度目录
            grad_dir = os.path.join(EXPERIMENT_DIR, optimizer_type, frequency, "gradients")
            os.makedirs(grad_dir, exist_ok=True)
    
    # 比较结果目录
    comparison_dir = os.path.join(EXPERIMENT_DIR, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    return EXPERIMENT_DIR

# 打印实验配置
def print_experiment_config():
    """打印当前实验配置"""
    print("\n" + "="*50)
    print(f"实验名称: {EXPERIMENT_NAME}_{TIMESTAMP}")
    print(f"实验目录: {EXPERIMENT_DIR}")
    print(f"训练轮数: {EPOCHS}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"学习率: {LEARNING_RATE}")
    print(f"权重衰减: {WEIGHT_DECAY}")
    print(f"SGD动量: {SGD_MOMENTUM}")
    print(f"SGD Nesterov加速: {SGD_NESTEROV}")
    print(f"设备: {DEVICE}")
    print(f"随机种子: {SEED}")
    print(f"检查点保存间隔: 每{CHECKPOINT_INTERVAL}个epoch")
    print(f"频率: {', '.join(FREQUENCIES)}")
    print("="*50 + "\n") 