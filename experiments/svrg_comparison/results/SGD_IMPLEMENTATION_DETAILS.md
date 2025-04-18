# SGD+Momentum 优化器实现细节

## 概述

本文档详细说明了为音频角度分类任务实现的SGD+Momentum优化器的技术细节。此优化器是经典随机梯度下降(SGD)的扩展版本，通过添加动量机制(Momentum)和支持Nesterov加速，旨在提高模型的收敛速度和性能。

## 核心组件

### 1. TrackedSGD 优化器类

继承了PyTorch的SGD优化器并扩展了其功能，主要实现在`optimization/tracked_sgd.py`中。

```python
class TrackedSGD(Optimizer):
    """
    实现带梯度追踪的SGD优化器，支持动量和Nesterov加速
    """
    
    def __init__(self, params, lr=0.01, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        """
        初始化优化器
        
        参数:
            params (iterable): 待优化参数
            lr (float): 学习率
            momentum (float): 动量因子 (默认: 0)
            dampening (float): 动量阻尼因子 (默认: 0)
            weight_decay (float): 权重衰减系数 (默认: 0)
            nesterov (bool): 是否启用Nesterov动量 (默认: False)
        """
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        super(TrackedSGD, self).__init__(params, defaults)
        
        # 记录梯度历史，用于追踪和分析
        self.tracked_gradients = []
        self.param_groups_snapshot = None
        
    def step(self, closure=None):
        """
        执行单步优化
        
        参数:
            closure (callable, optional): 重新评估模型并返回损失的闭包
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            # 收集当前批次的梯度
            batch_gradients = []
            
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                
                # 记录原始梯度(用于分析)
                if p.grad is not None:
                    batch_gradients.append(p.grad.data.clone())
                
                # 应用权重衰减
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                
                # 应用动量
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                
                # 更新参数
                p.data.add_(d_p, alpha=-group['lr'])
            
            # 存储批次梯度信息
            if batch_gradients:
                self.tracked_gradients.append(batch_gradients)
        
        return loss
    
    def save_param_groups_snapshot(self):
        """保存参数快照，用于SVRG比较研究"""
        self.param_groups_snapshot = copy.deepcopy(self.param_groups)
    
    def get_tracked_gradients(self):
        """返回记录的梯度历史"""
        return self.tracked_gradients
```

### 2. SGDMomentumTrainer 训练器类

定制开发的训练器，适配SGD+Momentum优化器的特定需求，实现在`trainers/sgd_momentum_trainer.py`中。

```python
class SGDMomentumTrainer(BaseTrainer):
    """
    使用带动量的SGD优化器的模型训练器
    """
    
    def __init__(self, model, train_loader, val_loader, config, logger=None):
        """
        初始化SGD+Momentum训练器
        
        参数:
            model: 要训练的神经网络模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            config: 配置参数字典
            logger: 日志记录器(可选)
        """
        super(SGDMomentumTrainer, self).__init__(model, train_loader, val_loader, config, logger)
        
        # 从配置中提取SGD特定参数
        self.lr = config.get('learning_rate', 0.01)
        self.momentum = config.get('momentum', 0.9)
        self.nesterov = config.get('nesterov', False)
        self.weight_decay = config.get('weight_decay', 0)
        
        # 初始化优化器
        self.optimizer = TrackedSGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            nesterov=self.nesterov,
            weight_decay=self.weight_decay
        )
        
        # 训练状态和指标追踪
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.gradients_norm_history = []
        
        # 学习率调度器(可选)
        if config.get('use_lr_scheduler', False):
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=config.get('lr_step_size', 10),
                gamma=config.get('lr_gamma', 0.1)
            )
        else:
            self.lr_scheduler = None
    
    def train_epoch(self):
        """执行单个训练周期"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        epoch_gradients_norm = []
        
        # 设置进度条
        pbar = tqdm(self.train_loader)
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # 将数据移至设备(CPU/GPU)
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # 清除之前的梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            
            # 记录梯度范数
            gradients_norm = self._compute_gradients_norm()
            epoch_gradients_norm.append(gradients_norm)
            
            # 参数更新
            self.optimizer.step()
            
            # 统计信息更新
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 更新进度条信息
            pbar.set_description(
                f'Train Loss: {total_loss/(batch_idx+1):.3f} | '
                f'Train Acc: {100.*correct/total:.2f}%'
            )
        
        # 计算并存储本轮指标
        epoch_loss = total_loss / len(self.train_loader)
        epoch_accuracy = 100. * correct / total
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_accuracy)
        self.gradients_norm_history.append(np.mean(epoch_gradients_norm))
        
        # 更新学习率调度(如果启用)
        if self.lr_scheduler:
            self.lr_scheduler.step()
        
        return epoch_loss, epoch_accuracy
    
    def _compute_gradients_norm(self):
        """计算当前梯度的L2范数均值"""
        total_norm = 0
        count = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                count += 1
        if count == 0:
            return 0
        return (total_norm / count) ** 0.5
    
    def validate(self):
        """在验证集上评估模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # 计算并存储验证指标
        val_loss = total_loss / len(self.val_loader)
        val_accuracy = 100. * correct / total
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_accuracy)
        
        return val_loss, val_accuracy
    
    def get_training_history(self):
        """返回完整训练历史数据"""
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'gradients_norm_history': self.gradients_norm_history
        }
```

## 参数配置

优化器需要以下配置参数，通过`experiments/svrg_comparison/config.json`提供：

```json
{
  "sgd_momentum": {
    "learning_rate": 0.01,
    "momentum": 0.9,
    "nesterov": true,
    "weight_decay": 0.0001,
    "use_lr_scheduler": false,
    "lr_step_size": 10,
    "lr_gamma": 0.1
  }
}
```

## 使用示例

以下是使用SGD+Momentum优化器训练模型的示例：

```python
from models import AudioAngleClassifier
from datasets import AudioDataset
from trainers import SGDMomentumTrainer
import torch

# 准备数据
train_dataset = AudioDataset("data/audio_500hz/train")
val_dataset = AudioDataset("data/audio_500hz/val")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

# 初始化模型
model = AudioAngleClassifier(input_size=1024, num_classes=37)

# 定义训练配置
config = {
    "learning_rate": 0.01,
    "momentum": 0.9,
    "nesterov": True,
    "weight_decay": 0.0001,
    "epochs": 30,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "criterion": torch.nn.CrossEntropyLoss()
}

# 创建训练器
trainer = SGDMomentumTrainer(model, train_loader, val_loader, config)

# 开始训练
for epoch in range(config["epochs"]):
    train_loss, train_acc = trainer.train_epoch()
    val_loss, val_acc = trainer.validate()
    print(f"Epoch {epoch+1}/{config['epochs']}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

# 获取训练历史
history = trainer.get_training_history()
```

## 技术设计考量

1. **梯度追踪**：
   - TrackedSGD优化器实现了梯度追踪机制，便于分析训练过程中梯度的变化
   - 梯度历史可用于可视化和比较不同优化器的行为差异

2. **动量机制**：
   - 支持标准动量和Nesterov加速动量
   - 动量参数可调，帮助模型跳出局部最小值并加速收敛

3. **与SVRG的兼容性**：
   - 添加了`save_param_groups_snapshot`方法，支持与SVRG的对比实验
   - 确保接口一致性，方便不同优化器间的切换

4. **学习率调度**：
   - 可选的学习率调度支持，使用StepLR实现
   - 帮助在训练后期调整学习率，提高最终模型性能

## 实现注意事项

1. **梯度裁剪**：
   - 当前版本没有内置梯度裁剪，如果梯度爆炸问题严重，可考虑添加

2. **内存管理**：
   - 梯度追踪会增加内存消耗，对于大型模型需谨慎使用
   - 可以设置最大记录步数限制以控制内存使用

3. **批量大小影响**：
   - SGD+Momentum对批量大小比SVRG和Adam更敏感
   - 建议使用较大批量以获得更平稳的优化轨迹

## 优化器特性与局限性

### 优势

1. **收敛稳定性**：
   - 动量机制使收敛路径更平滑，减少震荡
   - 特别适合具有较高噪声梯度的问题

2. **克服局部最小值**：
   - 动量累积帮助优化器跨越平坦区域和逃离局部最小值
   - Nesterov加速进一步增强这一能力

3. **内存效率**：
   - 相比Adam等优化器，内存消耗更低
   - 不需要为每个参数维护额外状态变量(仅需动量缓冲区)

### 局限性

1. **超参数敏感性**：
   - 性能对学习率和动量参数敏感
   - 需要更细致的超参数调优

2. **特征缩放依赖**：
   - 对输入特征的缩放不如自适应方法(如Adam)鲁棒
   - 建议提前进行数据归一化

3. **初始收敛较慢**：
   - 在训练初期，收敛速度可能慢于Adam
   - 需要更多的训练步数才能达到相似的性能

## 参考文献

1. Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013). "On the importance of initialization and momentum in deep learning." ICML.

2. Qian, N. (1999). "On the momentum term in gradient descent learning algorithms." Neural Networks, 12(1), 145-151.

3. Nesterov, Y. (1983). "A method for unconstrained convex minimization problem with the rate of convergence O(1/k^2)." Doklady ANSSSR, 269, 543-547. 