# 添加SGD+Momentum优化器与三方比较实验

## 变更概述

本PR实现了SGD+Momentum优化器，并将其与现有的SVRG和Adam优化器进行了系统性对比，为音频角度分类任务提供了更全面的优化器性能评估。

## 主要变更

1. **实现了TrackedSGD类**：在现有优化器框架基础上，实现了支持动量和Nesterov加速的SGD优化器，并添加了梯度追踪功能
2. **创建了SGDMomentumTrainer类**：为SGD+Momentum优化器开发了专用的训练器类
3. **更新了实验配置**：增加了SGD优化器的参数配置，并为结果添加了更清晰的组织结构
4. **修改了比较逻辑**：更新了绘图代码以支持三个优化器的性能对比，并增强了命令行接口
5. **添加了运行脚本**：创建了`run_sgd_comparison.sh`脚本，设置了推荐的参数并提供了便捷的实验入口
6. **完善了文档**：在README中添加了SGD+Momentum的详细说明和学术参考

## 实验结果摘要

在三种不同频率（500Hz、1000Hz、3000Hz）的音频数据上进行了对比实验，主要发现如下：

- **Adam优势**：训练早期收敛速度快，验证性能稳定，整体表现平衡
- **SGD+Momentum优势**：训练后期达到最高训练准确率，在某些频率下验证准确率可与Adam媲美
- **SVRG劣势**：收敛速度明显慢于其他两种优化器，且最终性能不如其他两种优化器

详细实验结果和分析请参见`experiments/svrg_comparison/results/THREE_OPTIMIZERS_REPORT.md`。

## 技术实现要点

### TrackedSGD优化器
- 基于PyTorch的SGD优化器扩展，添加了梯度追踪功能
- 支持动量（momentum）参数和Nesterov加速
- 实现了与SVRG兼容的接口，便于实验比较

### SGDMomentumTrainer
- 实现了与其他优化器训练器一致的训练逻辑和接口
- 支持动态学习率调整和梯度追踪
- 提供了详细的训练过程记录

## 使用方法

### 使用预设参数运行实验
```bash
./experiments/svrg_comparison/run_sgd_comparison.sh
```

### 自定义参数运行实验
```bash
python -m experiments.svrg_comparison.run_optimizer_comparison --frequencies 500hz 1000hz 3000hz --epochs 30 --sgd-momentum 0.9 --sgd-nesterov
```

## 后续工作

1. 为SGD+Momentum探索学习率调度策略
2. 对SGD超参数进行更细致的调优
3. 尝试在训练不同阶段混合使用不同优化器
4. 在更复杂的模型上测试各优化器的表现差异

## 相关文献

- Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013). On the importance of initialization and momentum in deep learning. ICML.
- Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv:1412.6980.
- Johnson, R., & Zhang, T. (2013). Accelerating stochastic gradient descent using predictive variance reduction. NIPS. 