# SVRG优化器比较实验

本目录包含了比较SVRG (Stochastic Variance Reduced Gradient) 优化器与标准的Adam优化器及SGD+Momentum优化器在音频角度分类任务上的性能差异的实验代码和结果。

## 目录结构

```
experiments/svrg_comparison/
├── data/                        # 实验数据缓存目录
├── models/                      # 实验特定模型定义
├── results/                     # 实验结果存储目录
│   ├── summary/                 # 实验结果汇总图表
│   ├── EXPERIMENT_REPORT.md     # 实验详细报告
│   └── svrg_comparison_*/       # 按时间戳组织的实验结果
├── utils/                       # 实验工具模块
│   ├── data_tracker.py          # 实验数据记录工具
│   ├── enhanced_optimizers.py   # 增强的优化器实现
│   ├── experiment_config.py     # 实验配置参数
│   └── trainers.py              # 训练器实现
├── README.md                    # 本文件
└── run_optimizer_comparison.py  # 实验入口脚本
```

## 实验背景

SVRG是一种旨在减少随机梯度方差的优化算法，通过周期性计算全梯度来优化收敛过程。本实验旨在对比SVRG与广泛使用的Adam优化器在音频角度分类任务上的表现差异。

## 运行实验

要运行实验，使用以下命令：

```bash
python -m experiments.svrg_comparison.run_optimizer_comparison [参数]
```

或者使用预配置的脚本：

```bash
# 运行包含SGD+Momentum的完整比较实验
./experiments/svrg_comparison/run_sgd_comparison.sh
```

### 参数说明

- `--frequencies`: 要训练的频率，可选 500hz, 1000hz, 3000hz, all (默认: 全部)
- `--material`: 材质类型 (默认: plastic)
- `--epochs`: 训练轮数 (默认: 30)
- `--batch-size`: 批次大小 (默认: 32)
- `--learning-rate`: 学习率 (默认: 0.001)
- `--weight-decay`: 权重衰减 (默认: 0.0001)
- `--sgd-momentum`: SGD动量系数 (默认: 0.9)
- `--sgd-nesterov`: 启用Nesterov动量加速
- `--checkpoint-interval`: 检查点保存间隔 (默认: 5)
- `--seed`: 随机种子 (默认: 42)
- `--device`: 计算设备 (cpu, cuda, mps, auto)
- `--only-standard`: 仅使用标准优化器(Adam)训练
- `--only-svrg`: 仅使用SVRG优化器训练
- `--only-sgd`: 仅使用SGD+Momentum优化器训练
- `--skip-training`: 跳过训练步骤，仅执行评估

### 示例

```bash
# 在所有频率上运行完整对比
python -m experiments.svrg_comparison.run_optimizer_comparison

# 仅在1000Hz频率上运行
python -m experiments.svrg_comparison.run_optimizer_comparison --frequencies 1000hz

# 仅测试Adam优化器
python -m experiments.svrg_comparison.run_optimizer_comparison --only-standard
```

## 结果查看

实验结果保存在 `results/` 目录中，按照实验运行的时间戳组织。每次实验会生成以下内容：

1. 模型检查点 (每5个epoch保存一次)
2. 训练和验证指标历史记录
3. 梯度统计信息
4. 批次和样本级别的详细记录
5. 两种优化器性能对比图表

详细的实验分析请查看 [实验报告](results/EXPERIMENT_REPORT.md)。

## 主要发现

基于实验结果，在音频角度分类任务上：

1. Adam优化器在收敛速度、验证准确率和训练稳定性方面整体优于SVRG优化器
2. SGD+Momentum在训练开始阶段收敛较慢，但最终性能可与Adam相当，有时甚至更佳
3. 随着频率增加，各优化器的性能差距有所减小
4. SVRG由于需要周期性计算全梯度，训练时间较长

更多详细分析请参考完整的实验报告。

## 参考资料

1. Johnson, R., & Zhang, T. (2013). Accelerating stochastic gradient descent using predictive variance reduction. NIPS.
2. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
3. Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013). On the importance of initialization and momentum in deep learning. ICML. 