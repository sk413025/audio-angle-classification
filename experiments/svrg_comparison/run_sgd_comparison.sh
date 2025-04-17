#!/bin/bash

# 运行包含SGD+Momentum的优化器比较实验

# 设置项目根目录
PROJECT_ROOT=$(pwd)

echo "========================================================"
echo "      SVRG vs Adam vs SGD+Momentum 优化器比较实验"
echo "========================================================"
echo "开始时间: $(date)"
echo "项目目录: $PROJECT_ROOT"
echo "========================================================"

# 创建日志目录
mkdir -p experiments/svrg_comparison/logs

# 脚本实际运行路径
LOG_FILE="experiments/svrg_comparison/logs/sgd_comparison_$(date +%Y%m%d_%H%M%S).log"

# 运行实验 - 使用所有三种优化器在全部频率上进行比较
# 参数设置:
# --frequencies: 要训练的频率
# --epochs: 训练的总轮数
# --checkpoint-interval: 检查点保存间隔
# --learning-rate: 优化器学习率
# --sgd-momentum: SGD动量参数
# --sgd-nesterov: 启用Nesterov动量
# --batch-size: 批次大小

python -m experiments.svrg_comparison.run_optimizer_comparison \
    --frequencies 500hz 1000hz 3000hz \
    --epochs 30 \
    --checkpoint-interval 5 \
    --learning-rate 0.001 \
    --sgd-momentum 0.9 \
    --sgd-nesterov \
    --batch-size 32 | tee "$LOG_FILE"

echo "========================================================"
echo "实验完成时间: $(date)"
echo "日志文件已保存至: $LOG_FILE"
echo "========================================================" 