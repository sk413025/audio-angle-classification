# SVRG、Adam与SGD+Momentum优化器性能比较实验报告

## 实验背景

深度学习优化算法的选择对模型训练效果具有重要影响。本实验对比了三种流行的优化算法在音频角度分类任务上的性能表现：

1. **Adam**：自适应学习率优化算法，结合了动量和RMSProp的优点，广泛应用于深度学习训练
2. **SVRG (Stochastic Variance Reduced Gradient)**：通过周期性计算全梯度来减少随机梯度的方差，理论上可加速收敛
3. **SGD+Momentum**：传统的随机梯度下降配合动量项，可以加速训练并帮助逃离局部最小值

通过对这三种优化器在不同频率数据上的表现进行系统比较，我们希望深入理解它们的优势与局限性，为音频角度分类任务选择最适合的优化策略。

## 实验目标

1. 评估三种优化器在不同频率数据上的收敛速度
2. 比较优化器在训练和验证准确率方面的最终性能
3. 分析优化器在不同频率（500Hz、1000Hz、3000Hz）下的表现差异
4. 探究各优化器的计算效率与训练稳定性

## 实验设置

- **数据集**：音频角度分类数据集（材质：plastic）
- **频率**：500Hz、1000Hz、3000Hz
- **模型**：SimpleCNNAudioRanker（相同架构用于所有优化器）
- **训练轮数**：30个epoch
- **批次大小**：32
- **优化器配置**：
  - Adam：学习率=0.001，权重衰减=1e-4
  - SVRG：学习率=0.001，权重衰减=1e-4
  - SGD+Momentum：学习率=0.001，动量=0.9，权重衰减=1e-4，Nesterov加速=True
- **损失函数**：MarginRankingLoss（margin=1.0）

## 实验结果

### 500Hz频率

![500Hz对比](svrg_comparison_20250417_182014/comparison/comparison_500hz_20250417_183427.png)

在500Hz频率数据上，实验结果表明：

- **训练损失**：Adam和SGD+Momentum收敛速度相近，且明显快于SVRG；SGD+Momentum在后期达到最低训练损失
- **验证损失**：Adam的验证损失整体最低，SGD+Momentum次之，SVRG表现相对较差
- **训练准确率**：SGD+Momentum最终达到约87%，高于Adam的83%和SVRG的72%
- **验证准确率**：Adam最终准确率达到约85%，SGD+Momentum约81%，而SVRG仅达到约79%

### 1000Hz频率

![1000Hz对比](svrg_comparison_20250417_182014/comparison/comparison_1000hz_20250417_184927.png)

在1000Hz频率数据上：

- **训练损失**：Adam早期收敛最快，但最终SGD+Momentum达到最低训练损失
- **验证损失**：Adam表现最佳，SGD+Momentum接近但稍逊，SVRG明显偏高
- **训练准确率**：SGD+Momentum最终达到约88%，高于Adam的约84%和SVRG的约75%
- **验证准确率**：SGD+Momentum和Adam最终表现相当，都达到约89%，而SVRG约为82%

### 3000Hz频率

![3000Hz对比](svrg_comparison_20250417_182014/comparison/comparison_3000hz_20250417_190531.png)

在3000Hz频率数据上：

- **训练损失**：SGD+Momentum表现最佳，达到最低训练损失值约0.44
- **验证损失**：三种优化器差异缩小，最终Adam表现略好
- **训练准确率**：SGD+Momentum最终达到约85%，高于Adam的78%和SVRG的73%
- **验证准确率**：Adam最终略高，约53%；SGD+Momentum约49%；SVRG约48%

## 分析与讨论

1. **收敛速度比较**：
   - Adam在所有频率下早期收敛速度都较快，特别是在验证集上表现稳定
   - SGD+Momentum开始收敛较慢，但随着训练进行，在训练损失上超过了其他两种优化器
   - SVRG的收敛速度明显慢于其他两种优化器，且需要更多的计算资源

2. **最终准确率比较**：
   - 在验证准确率方面，Adam在500Hz和3000Hz上表现最佳，而在1000Hz上与SGD+Momentum旗鼓相当
   - SGD+Momentum在训练准确率上表现最好，但存在一定过拟合风险
   - SVRG在所有频率上的准确率均落后于其他两种优化器

3. **训练稳定性**：
   - Adam训练过程最稳定，验证指标波动小
   - SGD+Momentum在训练后期出现了一些震荡，特别是在验证损失上
   - SVRG训练过程波动较大，特别是在验证指标上

4. **频率影响**：
   - 随着频率增加，三种优化器在验证准确率上的表现差距缩小
   - 在3000Hz上，三种优化器的验证准确率都有所下降，可能与高频特征对分类任务的贡献较小有关

5. **计算效率**：
   - SVRG由于需要定期计算全梯度，训练时间显著长于其他两种优化器
   - Adam和SGD+Momentum的计算效率相近，但SGD+Momentum在某些批次上计算速度略快

## 结论

基于本次实验结果，在音频角度分类任务上的优化器选择可得出以下结论：

1. **Adam优势**：训练早期收敛快，验证性能优秀，训练过程稳定。适合对验证性能和训练稳定性要求高的场景。

2. **SGD+Momentum优势**：训练后期可达到最高训练准确率，在某些频率上的验证准确率与Adam相当。适合可以进行更长时间训练，且有足够计算资源调优的场景。

3. **SVRG局限性**：在该任务上，SVRG并未展现出其理论上的优势，收敛速度慢且最终性能不如其他优化器。并且计算开销大，训练效率低。

综合考虑，对于音频角度分类任务，**Adam优化器**在效率和性能平衡方面表现最佳，是首选的优化算法。如果追求更高的训练准确率，并有足够资源进行充分训练，**SGD+Momentum**也是一个很好的选择。

## 下一步计划

1. **学习率调度策略探索**：为SGD+Momentum尝试更复杂的学习率调度策略，如余弦退火（Cosine Annealing）
2. **超参数调优**：对SGD+Momentum的动量系数和Nesterov加速进行更细致的调优
3. **混合优化策略**：尝试在训练不同阶段使用不同优化器，如早期使用Adam快速收敛，后期切换到SGD+Momentum微调
4. **更深层模型测试**：在更复杂的神经网络上测试各优化器的表现
5. **梯度分析**：对三种优化器的梯度分布和变化进行深入分析，探寻性能差异的原因

通过上述计划，我们希望能够更全面地理解不同优化器的特性，并开发出更高效的优化策略用于音频角度分类任务。

## 参考资料

1. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
2. Johnson, R., & Zhang, T. (2013). Accelerating stochastic gradient descent using predictive variance reduction. NIPS.
3. Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013). On the importance of initialization and momentum in deep learning. ICML. 