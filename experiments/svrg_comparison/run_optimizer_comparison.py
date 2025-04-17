#!/usr/bin/env python
"""
SVRG优化器比较实验

本脚本用于执行SVRG优化器与标准优化器(Adam)的性能比较实验。

用法:
    python experiments/svrg_comparison/run_optimizer_comparison.py [--options]

例子:
    python experiments/svrg_comparison/run_optimizer_comparison.py --frequencies 500hz 1000hz
"""

import os
import sys
import argparse
import torch
import pickle
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加项目根目录到路径，确保导入正确
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入自定义模块
from datasets import SpectrogramDatasetWithMaterial, RankingPairDataset
from models.resnet_ranker import SimpleCNNAudioRanker
from utils.common_utils import worker_init_fn, set_seed
from torch.nn import MarginRankingLoss
import config as project_config

# 导入实验模块
from experiments.svrg_comparison.utils.experiment_config import (
    EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, CHECKPOINT_INTERVAL,
    SEED, DEVICE, FREQUENCIES, EXPERIMENT_DIR, LOG_INTERVAL, SGD_MOMENTUM, SGD_NESTEROV,
    create_experiment_dirs, print_experiment_config
)
from experiments.svrg_comparison.utils.trainers import StandardOptimTrainer, SVRGTrainer, SGDMomentumTrainer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="SVRG优化器性能比较实验")
    
    parser.add_argument('--frequencies', nargs='+', type=str, default=FREQUENCIES,
                       choices=['500hz', '1000hz', '3000hz', 'all'],
                       help='要训练的频率 (默认: 全部)')
    
    parser.add_argument('--material', type=str, default=project_config.MATERIAL,
                       help=f'材质类型 (默认: {project_config.MATERIAL})')
    
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                       help=f'训练轮数 (默认: {EPOCHS})')
    
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help=f'批次大小 (默认: {BATCH_SIZE})')
    
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE,
                       help=f'学习率 (默认: {LEARNING_RATE})')
    
    parser.add_argument('--weight-decay', type=float, default=WEIGHT_DECAY,
                       help=f'权重衰减 (默认: {WEIGHT_DECAY})')
    
    parser.add_argument('--checkpoint-interval', type=int, default=CHECKPOINT_INTERVAL,
                       help=f'检查点保存间隔 (默认: {CHECKPOINT_INTERVAL})')
    
    parser.add_argument('--seed', type=int, default=SEED,
                       help=f'随机种子 (默认: {SEED})')
    
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps', 'auto'],
                       default='auto',
                       help='计算设备 (默认: auto)')
    
    parser.add_argument('--only-standard', action='store_true',
                       help='仅使用标准优化器(Adam)训练')
    
    parser.add_argument('--only-svrg', action='store_true',
                       help='仅使用SVRG优化器训练')
    
    parser.add_argument('--only-sgd', action='store_true',
                       help='仅使用SGD+Momentum优化器训练')
    
    parser.add_argument('--skip-training', action='store_true',
                       help='跳过训练步骤，仅执行评估（需要已有训练结果）')
    
    parser.add_argument('--sgd-momentum', type=float, default=SGD_MOMENTUM,
                       help=f'SGD动量系数 (默认: {SGD_MOMENTUM})')
    
    parser.add_argument('--sgd-nesterov', action='store_true', default=SGD_NESTEROV,
                       help='是否使用Nesterov动量')
    
    args = parser.parse_args()
    
    # 处理'all'频率选项
    if 'all' in args.frequencies or len(args.frequencies) == 0:
        args.frequencies = FREQUENCIES
    
    # 设置设备
    if args.device == 'auto':
        args.device = DEVICE
    else:
        if args.device == 'cuda' and not torch.cuda.is_available():
            print("CUDA不可用，使用CPU")
            args.device = torch.device('cpu')
        elif args.device == 'mps' and not (hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            print("MPS不可用，使用CPU")
            args.device = torch.device('cpu')
        else:
            args.device = torch.device(args.device)
    
    return args


def prepare_data(frequency, material, batch_size, seed):
    """
    准备数据加载器
    
    Args:
        frequency (str): 频率
        material (str): 材质
        batch_size (int): 批次大小
        seed (int): 随机种子
        
    Returns:
        tuple: (训练数据加载器, 验证数据加载器)
    """
    print(f"加载数据集: 频率={frequency}, 材质={material}")
    
    try:
        dataset = SpectrogramDatasetWithMaterial(
            project_config.DATA_ROOT,
            project_config.CLASSES,
            project_config.SEQ_NUMS,
            frequency,
            material
        )
    except Exception as e:
        print(f"加载数据集出错: {e}")
        return None, None
    
    if len(dataset) == 0:
        print("数据集为空，无法训练")
        return None, None
    
    # 拆分数据集
    train_size = int(0.70 * len(dataset))
    val_size = len(dataset) - train_size
    
    if train_size < 4 or val_size < 4:
        print(f"数据集过小 (总计: {len(dataset)})，无法进行训练/验证拆分")
        return None, None
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    # 创建排序数据集
    train_ranking_dataset = RankingPairDataset(train_dataset)
    val_ranking_dataset = RankingPairDataset(val_dataset)
    
    # 创建数据加载器
    device_type = DEVICE.type if hasattr(DEVICE, 'type') else str(DEVICE)
    num_workers = 0 if device_type == 'mps' else 4
    
    train_loader = torch.utils.data.DataLoader(
        train_ranking_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(seed)
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_ranking_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(seed)
    )
    
    return train_loader, val_loader


def save_experiment_metadata(args):
    """保存实验元数据"""
    # 将args复制一份，避免修改原始对象
    args_dict = vars(args).copy()
    
    # 确保device能够序列化为JSON
    if 'device' in args_dict and not isinstance(args_dict['device'], str):
        args_dict['device'] = str(args_dict['device'])
    
    metadata = {
        'args': args_dict,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'device': str(args.device),
        'frequencies': args.frequencies,
        'pytorch_version': torch.__version__
    }
    
    metadata_path = os.path.join(EXPERIMENT_DIR, "experiment_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"实验元数据已保存至: {metadata_path}")


def train_with_optimizer(optimizer_type, frequency, args):
    """
    使用指定优化器进行训练
    
    Args:
        optimizer_type (str): 优化器类型 ('standard', 'svrg' 或 'sgd_momentum')
        frequency (str): 频率
        args: 命令行参数
        
    Returns:
        tuple: (模型, 训练历史)
    """
    print(f"\n{'='*20} 训练 {optimizer_type} 优化器 - 频率 {frequency} {'='*20}")
    
    # 准备数据
    train_loader, val_loader = prepare_data(
        frequency=frequency,
        material=args.material,
        batch_size=args.batch_size,
        seed=args.seed
    )
    
    if train_loader is None or val_loader is None:
        print(f"没有数据用于训练 - 频率: {frequency}")
        return None, None
    
    # 确定数据尺寸
    spec_shape = train_loader.dataset.dataset.dataset.data.shape[2]
    
    # 初始化模型
    model = SimpleCNNAudioRanker(n_freqs=spec_shape)
    model.to(args.device)
    
    # 初始化损失函数
    criterion = MarginRankingLoss(margin=1.0).to(args.device)
    
    # 初始化训练器
    if optimizer_type == 'standard':
        trainer = StandardOptimTrainer(
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            device=args.device,
            save_dir=EXPERIMENT_DIR,
            frequency=frequency,
            checkpoint_interval=args.checkpoint_interval,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif optimizer_type == 'sgd_momentum':
        trainer = SGDMomentumTrainer(
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            device=args.device,
            save_dir=EXPERIMENT_DIR,
            frequency=frequency,
            checkpoint_interval=args.checkpoint_interval,
            lr=args.learning_rate,
            momentum=args.sgd_momentum,
            weight_decay=args.weight_decay,
            nesterov=args.sgd_nesterov
        )
    else:  # svrg
        trainer = SVRGTrainer(
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            device=args.device,
            save_dir=EXPERIMENT_DIR,
            frequency=frequency,
            checkpoint_interval=args.checkpoint_interval,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    
    # 训练模型
    print(f"开始训练 - 优化器: {optimizer_type}, 频率: {frequency}")
    history = trainer.train(epochs=args.epochs, log_interval=LOG_INTERVAL)
    
    # 保存训练历史
    history_path = os.path.join(
        EXPERIMENT_DIR,
        optimizer_type,
        frequency,
        f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    )
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    
    print(f"训练历史已保存至: {history_path}")
    
    return model, history


def plot_comparison(frequency):
    """
    绘制多种优化器的性能比较图
    
    Args:
        frequency (str): 频率
    """
    # 查找训练历史文件
    standard_dir = os.path.join(EXPERIMENT_DIR, 'standard', frequency)
    svrg_dir = os.path.join(EXPERIMENT_DIR, 'svrg', frequency)
    sgd_momentum_dir = os.path.join(EXPERIMENT_DIR, 'sgd_momentum', frequency)
    
    standard_history_files = [f for f in os.listdir(standard_dir) if f.startswith('training_history_') and f.endswith('.pkl')]
    svrg_history_files = [f for f in os.listdir(svrg_dir) if f.startswith('training_history_') and f.endswith('.pkl')]
    sgd_momentum_history_files = [f for f in os.listdir(sgd_momentum_dir) if f.startswith('training_history_') and f.endswith('.pkl')]
    
    if not standard_history_files or not svrg_history_files or not sgd_momentum_history_files:
        print(f"没有足够的历史数据进行比较 - 频率: {frequency}")
        return
    
    # 加载最新的历史数据
    standard_history_file = sorted(standard_history_files)[-1]
    svrg_history_file = sorted(svrg_history_files)[-1]
    sgd_momentum_history_file = sorted(sgd_momentum_history_files)[-1]
    
    with open(os.path.join(standard_dir, standard_history_file), 'rb') as f:
        standard_history = pickle.load(f)
    
    with open(os.path.join(svrg_dir, svrg_history_file), 'rb') as f:
        svrg_history = pickle.load(f)
    
    with open(os.path.join(sgd_momentum_dir, sgd_momentum_history_file), 'rb') as f:
        sgd_momentum_history = pickle.load(f)
    
    # 创建比较目录
    comparison_dir = os.path.join(EXPERIMENT_DIR, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # 绘制训练损失比较
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(standard_history['epochs'], standard_history['train_loss'], 'b-', label='Adam')
    plt.plot(svrg_history['epochs'], svrg_history['train_loss'], 'r-', label='SVRG')
    plt.plot(sgd_momentum_history['epochs'], sgd_momentum_history['train_loss'], 'g-', label='SGD+Momentum')
    plt.title(f'训练损失比较 - {frequency}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制验证损失比较
    plt.subplot(2, 2, 2)
    plt.plot(standard_history['epochs'], standard_history['val_loss'], 'b-', label='Adam')
    plt.plot(svrg_history['epochs'], svrg_history['val_loss'], 'r-', label='SVRG')
    plt.plot(sgd_momentum_history['epochs'], sgd_momentum_history['val_loss'], 'g-', label='SGD+Momentum')
    plt.title(f'验证损失比较 - {frequency}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制训练准确率比较
    plt.subplot(2, 2, 3)
    plt.plot(standard_history['epochs'], standard_history['train_accuracy'], 'b-', label='Adam')
    plt.plot(svrg_history['epochs'], svrg_history['train_accuracy'], 'r-', label='SVRG')
    plt.plot(sgd_momentum_history['epochs'], sgd_momentum_history['train_accuracy'], 'g-', label='SGD+Momentum')
    plt.title(f'训练准确率比较 - {frequency}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # 绘制验证准确率比较
    plt.subplot(2, 2, 4)
    plt.plot(standard_history['epochs'], standard_history['val_accuracy'], 'b-', label='Adam')
    plt.plot(svrg_history['epochs'], svrg_history['val_accuracy'], 'r-', label='SVRG')
    plt.plot(sgd_momentum_history['epochs'], sgd_momentum_history['val_accuracy'], 'g-', label='SGD+Momentum')
    plt.title(f'验证准确率比较 - {frequency}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # 保存图表
    comparison_path = os.path.join(comparison_dir, f"comparison_{frequency}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(comparison_path)
    plt.close()
    
    print(f"比较图表已保存至: {comparison_path}")


def main():
    """主函数"""
    args = parse_args()
    
    # 创建实验目录
    create_experiment_dirs()
    
    # 打印实验配置
    print_experiment_config()
    
    # 保存实验元数据
    save_experiment_metadata(args)
    
    # 设置随机种子
    set_seed(args.seed)
    print(f"使用随机种子: {args.seed}")
    
    # 遍历每个频率
    for frequency in args.frequencies:
        if not args.skip_training:
            if not args.only_svrg and not args.only_sgd:
                # 使用标准优化器(Adam)训练
                standard_model, standard_history = train_with_optimizer(
                    optimizer_type='standard',
                    frequency=frequency,
                    args=args
                )
            
            if not args.only_standard and not args.only_sgd:
                # 使用SVRG优化器训练
                svrg_model, svrg_history = train_with_optimizer(
                    optimizer_type='svrg',
                    frequency=frequency,
                    args=args
                )
                
            if not args.only_standard and not args.only_svrg:
                # 使用SGD+Momentum优化器训练
                sgd_model, sgd_history = train_with_optimizer(
                    optimizer_type='sgd_momentum',
                    frequency=frequency,
                    args=args
                )
        
        # 绘制比较图表
        if (not args.only_standard and not args.only_svrg and not args.only_sgd) or args.skip_training:
            plot_comparison(frequency)
    
    print("\n所有实验已完成！")
    print(f"实验结果保存在: {EXPERIMENT_DIR}")


if __name__ == "__main__":
    main() 