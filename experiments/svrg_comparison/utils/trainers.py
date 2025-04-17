"""
训练器模块
实现不同优化器的训练流程
"""

import os
import torch
import time
import numpy as np
from tqdm import tqdm
from datetime import datetime
import logging

from experiments.svrg_comparison.utils.data_tracker import BatchRecorder, GradientTracker, ModelCheckpointer
from experiments.svrg_comparison.utils.enhanced_optimizers import TrackedAdam, TrackedSVRG, TrackedSVRGSnapshot, TrackedSGD


class BaseTrainer:
    """
    基础训练器类
    提供通用的训练功能
    """
    
    def __init__(self, 
                model, 
                criterion, 
                train_loader, 
                val_loader, 
                device, 
                save_dir,
                optimizer_type,
                frequency,
                checkpoint_interval=5):
        """
        初始化基础训练器
        
        Args:
            model: 要训练的模型
            criterion: 损失函数
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            device: 训练设备(CPU/GPU/MPS)
            save_dir: 保存结果的目录
            optimizer_type: 优化器类型 ('svrg' 或 'standard')
            frequency: 频率 (如 '500hz')
            checkpoint_interval: 保存检查点的间隔（每隔多少个epoch）
        """
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 初始化追踪器
        self.batch_recorder = BatchRecorder(
            save_dir=save_dir,
            optimizer_type=optimizer_type,
            frequency=frequency,
            save_full_batch_interval=checkpoint_interval
        )
        
        self.gradient_tracker = GradientTracker(
            model=model,
            save_dir=save_dir,
            optimizer_type=optimizer_type,
            frequency=frequency
        )
        
        self.model_checkpointer = ModelCheckpointer(
            save_dir=save_dir,
            optimizer_type=optimizer_type,
            frequency=frequency,
            save_interval=checkpoint_interval
        )
        
        # 设置日志
        self.logger = logging.getLogger(f"Trainer_{optimizer_type}_{frequency}")
        self.logger.setLevel(logging.INFO)
        
        # 创建文件处理器
        log_dir = os.path.join(save_dir, optimizer_type, frequency)
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # 添加处理器到日志记录器
        self.logger.addHandler(file_handler)
        
        # 训练指标记录
        self.history = {
            'optimizer_type': optimizer_type,
            'frequency': frequency,
            'epochs': [],
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': [],
            'epoch_times': []
        }
        
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        
        self.logger.info(f"初始化训练器: {optimizer_type} - {frequency}")
    
    def train_epoch(self, epoch):
        """
        训练一个epoch
        
        Args:
            epoch (int): 当前epoch编号
            
        Returns:
            dict: 训练统计信息
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def validate(self, epoch):
        """
        在验证集上评估模型
        
        Args:
            epoch (int): 当前epoch编号
            
        Returns:
            dict: 验证统计信息
        """
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_idx, (data1, data2, targets, label1, label2) in enumerate(self.val_loader):
                data1, data2, targets = data1.to(self.device), data2.to(self.device), targets.to(self.device)
                
                outputs1 = self.model(data1).view(-1)
                outputs2 = self.model(data2).view(-1)
                targets = targets.view(-1)
                
                loss = self.criterion(outputs1, outputs2, targets)
                
                val_loss += loss.item()
                
                predictions = (outputs1 > outputs2) == (targets > 0)
                val_correct += predictions.sum().item()
                val_total += targets.size(0)
                
                # 记录批次数据
                self.batch_recorder.record_batch(
                    epoch=epoch,
                    batch_idx=batch_idx,
                    data1=data1,
                    data2=data2,
                    targets=targets,
                    label1=label1,
                    label2=label2,
                    outputs1=outputs1,
                    outputs2=outputs2,
                    loss=loss
                )
        
        val_loss = val_loss / len(self.val_loader) if len(self.val_loader) > 0 else float('inf')
        val_accuracy = 100.0 * val_correct / val_total if val_total > 0 else 0.0
        
        # 记录验证指标
        val_metrics = {
            'loss': val_loss,
            'accuracy': val_accuracy,
            'correct': val_correct,
            'total': val_total
        }
        
        # 检查是否是最佳模型
        is_best = False
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            is_best = True
            self.logger.info(f"Epoch {epoch}: 新的最佳验证损失 {val_loss:.4f}")
        
        if val_accuracy > self.best_val_acc:
            self.best_val_acc = val_accuracy
            self.logger.info(f"Epoch {epoch}: 新的最佳验证准确率 {val_accuracy:.2f}%")
        
        return val_metrics, is_best
    
    def train(self, epochs, log_interval=10):
        """
        训练模型
        
        Args:
            epochs (int): 训练的总epoch数
            log_interval (int): 日志输出间隔（每隔多少个batch输出一次）
            
        Returns:
            dict: 训练历史记录
        """
        self.logger.info(f"开始训练，总共 {epochs} 个epochs")
        
        for epoch in range(epochs):
            self.logger.info(f"\n===== Epoch {epoch+1}/{epochs} =====")
            epoch_start_time = time.time()
            
            # 训练一个epoch
            train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_metrics, is_best = self.validate(epoch)
            
            # 计算epoch耗时
            epoch_time = time.time() - epoch_start_time
            
            # 更新历史记录
            self.history['epochs'].append(epoch + 1)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_accuracy'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['epoch_times'].append(epoch_time)
            
            # 获取学习率
            current_lr = self._get_current_lr()
            self.history['learning_rate'].append(current_lr)
            
            # 保存epoch摘要
            self.batch_recorder.save_epoch_summary(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                learning_rate=current_lr
            )
            
            # 保存模型检查点
            self.model_checkpointer.save_checkpoint(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                metrics={
                    'train': train_metrics,
                    'val': val_metrics
                },
                is_best=is_best
            )
            
            # 打印epoch摘要
            self._print_epoch_summary(epoch, epochs, train_metrics, val_metrics, epoch_time, current_lr)
        
        # 训练结束，保存样本历史记录
        self.batch_recorder.save_sample_histories()
        self.logger.info("训练完成")
        
        return self.history
    
    def _get_current_lr(self):
        """获取当前学习率"""
        if hasattr(self.optimizer, 'get_lr'):
            return self.optimizer.get_lr()
        
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        
        return None
    
    def _print_epoch_summary(self, epoch, epochs, train_metrics, val_metrics, epoch_time, lr):
        """打印epoch摘要信息"""
        print(f"\nEpoch {epoch+1}/{epochs} Summary")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Learning rate: {lr:.6f}")
        print(f"  Training   - Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.2f}%")
        print(f"  Validation - Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.2f}%")
        
        self.logger.info(f"Epoch {epoch+1} Summary - Time: {epoch_time:.2f}s, LR: {lr:.6f}")
        self.logger.info(f"  Training   - Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.2f}%")
        self.logger.info(f"  Validation - Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.2f}%")


class StandardOptimTrainer(BaseTrainer):
    """
    使用标准Adam优化器的训练器
    """
    
    def __init__(self, 
                model, 
                criterion, 
                train_loader, 
                val_loader, 
                device, 
                save_dir,
                frequency,
                checkpoint_interval=5,
                lr=0.001,
                weight_decay=1e-4):
        super(StandardOptimTrainer, self).__init__(
            model, 
            criterion, 
            train_loader, 
            val_loader, 
            device, 
            save_dir,
            'standard',  # 优化器类型固定为'standard'
            frequency,
            checkpoint_interval
        )
        
        # 初始化优化器
        self.optimizer = TrackedAdam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # 初始化学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            verbose=True
        )
        
        self.logger.info(f"初始化Adam优化器 - lr={lr}, weight_decay={weight_decay}")
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # 使用tqdm创建进度条
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (data1, data2, targets, label1, label2) in enumerate(progress_bar):
            data1, data2, targets = data1.to(self.device), data2.to(self.device), targets.to(self.device)
            
            # 清除梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs1 = self.model(data1).view(-1)
            outputs2 = self.model(data2).view(-1)
            targets = targets.view(-1)
            
            # 计算损失
            loss = self.criterion(outputs1, outputs2, targets)
            
            # 反向传播
            loss.backward()
            
            # 计算梯度统计信息
            grad_stats = self.optimizer.step(compute_stats=True)
            
            # 更新梯度统计信息记录
            if batch_idx % 10 == 0:  # 每10个批次记录一次
                self.gradient_tracker.save_gradient_stats(epoch, batch_idx, grad_stats)
            
            # 更新训练统计信息
            train_loss += loss.item()
            
            # 计算准确率
            predictions = (outputs1 > outputs2) == (targets > 0)
            train_correct += predictions.sum().item()
            train_total += targets.size(0)
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * predictions.sum().item() / targets.size(0)
            })
            
            # 记录批次数据
            self.batch_recorder.record_batch(
                epoch=epoch,
                batch_idx=batch_idx,
                data1=data1,
                data2=data2,
                targets=targets,
                label1=label1,
                label2=label2,
                outputs1=outputs1,
                outputs2=outputs2,
                loss=loss,
                gradients=grad_stats,
                model=self.model
            )
        
        # 计算平均训练损失和准确率
        train_loss = train_loss / len(self.train_loader) if len(self.train_loader) > 0 else float('inf')
        train_accuracy = 100.0 * train_correct / train_total if train_total > 0 else 0.0
        
        # 更新学习率调度器
        self.scheduler.step(train_loss)
        
        return {
            'loss': train_loss,
            'accuracy': train_accuracy,
            'correct': train_correct,
            'total': train_total
        }


class SVRGTrainer(BaseTrainer):
    """
    使用SVRG优化器的训练器
    """
    
    def __init__(self, 
                model, 
                criterion, 
                train_loader, 
                val_loader, 
                device, 
                save_dir,
                frequency,
                checkpoint_interval=5,
                lr=0.001,
                weight_decay=1e-4):
        """初始化SVRG优化器训练器"""
        super(SVRGTrainer, self).__init__(
            model, 
            criterion, 
            train_loader, 
            val_loader, 
            device, 
            save_dir,
            optimizer_type="svrg",
            frequency=frequency,
            checkpoint_interval=checkpoint_interval
        )
        
        # 初始化SVRG优化器
        self.optimizer = TrackedSVRG(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            name="SVRG"
        )
        
        # 初始化SVRG快照
        self.snapshot = TrackedSVRGSnapshot(model.parameters())
        
        # 初始化学习率
        self.lr = lr
        
        self.logger.info(f"使用SVRG优化器 - 学习率: {lr}, 权重衰减: {weight_decay}")
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # 计算全梯度用于快照
        self.model.train()
        self.snapshot.set_param_groups(self.optimizer.get_param_groups())
        self.model.zero_grad()
        
        full_grad_loss = 0.0
        self.logger.info("计算全梯度用于快照...")
        
        # 使用tqdm创建全梯度计算进度条
        snapshot_progress = tqdm(self.train_loader, desc="计算全梯度")
        
        # 计算所有样本的梯度（快照）
        for data1, data2, targets, label1, label2 in snapshot_progress:
            data1, data2, targets = data1.to(self.device), data2.to(self.device), targets.to(self.device)
            
            outputs1 = self.model(data1).view(-1)
            outputs2 = self.model(data2).view(-1)
            targets = targets.view(-1)
            
            loss = self.criterion(outputs1, outputs2, targets)
            loss.backward()
            full_grad_loss += loss.item()
        
        full_grad_loss /= len(self.train_loader)
        self.logger.info(f"全梯度损失: {full_grad_loss:.4f}")
        
        # 设置快照梯度到优化器
        self.optimizer.set_u(self.snapshot.get_param_groups())
        
        # SVRG训练步骤
        self.model.train()
        
        # 使用tqdm创建训练进度条
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (data1, data2, targets, label1, label2) in enumerate(progress_bar):
            data1, data2, targets = data1.to(self.device), data2.to(self.device), targets.to(self.device)
            
            # 重置快照模型参数为当前模型参数
            self.snapshot.set_param_groups(self.optimizer.get_param_groups())
            
            # 主模型的前向传播
            outputs1 = self.model(data1).view(-1)
            outputs2 = self.model(data2).view(-1)
            targets = targets.view(-1)
            
            # 计算主损失
            loss = self.criterion(outputs1, outputs2, targets)
            
            # 主模型的反向传播
            self.model.zero_grad()
            loss.backward()
            
            # 计算快照模型在相同数据上的梯度
            snapshot_outputs1 = self.model(data1).view(-1)
            snapshot_outputs2 = self.model(data2).view(-1)
            
            snapshot_loss = self.criterion(snapshot_outputs1, snapshot_outputs2, targets)
            
            # 快照的反向传播
            self.snapshot.zero_grad()
            snapshot_loss.backward()
            
            # 使用SVRG更新权重并计算梯度统计信息
            grad_stats = self.optimizer.step(self.snapshot.get_param_groups(), compute_stats=True)
            
            # 更新梯度统计信息记录
            if batch_idx % 10 == 0:  # 每10个批次记录一次
                self.gradient_tracker.save_gradient_stats(epoch, batch_idx, grad_stats)
            
            # 更新训练统计信息
            train_loss += loss.item()
            
            # 计算准确率
            predictions = (outputs1 > outputs2) == (targets > 0)
            train_correct += predictions.sum().item()
            train_total += targets.size(0)
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * predictions.sum().item() / targets.size(0)
            })
            
            # 记录批次数据
            self.batch_recorder.record_batch(
                epoch=epoch,
                batch_idx=batch_idx,
                data1=data1,
                data2=data2,
                targets=targets,
                label1=label1,
                label2=label2,
                outputs1=outputs1,
                outputs2=outputs2,
                loss=loss,
                gradients=grad_stats,
                model=self.model
            )
        
        # 计算平均训练损失和准确率
        train_loss = train_loss / len(self.train_loader) if len(self.train_loader) > 0 else float('inf')
        train_accuracy = 100.0 * train_correct / train_total if train_total > 0 else 0.0
        
        # 根据训练损失可以调整学习率（简单的学习率衰减）
        if epoch > 0 and epoch % 5 == 0:
            self.lr *= 0.5
            for group in self.optimizer.param_groups:
                group['lr'] = self.lr
            self.logger.info(f"Epoch {epoch+1}: 学习率降低至 {self.lr:.6f}")
        
        return {
            'loss': train_loss,
            'accuracy': train_accuracy,
            'correct': train_correct,
            'total': train_total
        } 


class SGDMomentumTrainer(BaseTrainer):
    """
    使用SGD+Momentum优化器的训练器
    """
    
    def __init__(self, 
                model, 
                criterion, 
                train_loader, 
                val_loader, 
                device, 
                save_dir,
                frequency,
                checkpoint_interval=5,
                lr=0.01,
                momentum=0.9,
                weight_decay=1e-4,
                nesterov=False):
        super(SGDMomentumTrainer, self).__init__(
            model, 
            criterion, 
            train_loader, 
            val_loader, 
            device, 
            save_dir,
            'sgd_momentum',  # 优化器类型固定为'sgd_momentum'
            frequency,
            checkpoint_interval
        )
        
        # 初始化优化器
        self.optimizer = TrackedSGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
        
        # 初始化学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            verbose=True
        )
        
        self.logger.info(f"初始化SGD+Momentum优化器 - lr={lr}, momentum={momentum}, weight_decay={weight_decay}, nesterov={nesterov}")
    
    def train_epoch(self, epoch):
        """
        训练一个epoch
        
        Args:
            epoch (int): 当前epoch编号
            
        Returns:
            dict: 训练统计信息
        """
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        epoch_start_time = time.time()
        
        for batch_idx, (data1, data2, targets, label1, label2) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}")):
            data1, data2, targets = data1.to(self.device), data2.to(self.device), targets.to(self.device)
            
            # 清除梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs1 = self.model(data1).view(-1)
            outputs2 = self.model(data2).view(-1)
            targets = targets.view(-1)
            
            # 计算损失
            loss = self.criterion(outputs1, outputs2, targets)
            
            # 反向传播
            loss.backward()
            
            # 计算梯度统计信息（如果需要）
            if epoch % 5 == 0 and batch_idx % 10 == 0:
                grad_stats = self.optimizer.step(compute_stats=True, record_grad=True)
                self.gradient_tracker.save_gradient_stats(epoch, batch_idx, grad_stats)
            else:
                self.optimizer.step()
            
            # 累计统计信息
            train_loss += loss.item()
            
            predictions = (outputs1 > outputs2) == (targets > 0)
            train_correct += predictions.sum().item()
            train_total += targets.size(0)
            
            # 记录批次数据
            self.batch_recorder.record_batch(
                epoch=epoch,
                batch_idx=batch_idx,
                data1=data1,
                data2=data2,
                targets=targets,
                label1=label1,
                label2=label2,
                outputs1=outputs1,
                outputs2=outputs2,
                loss=loss
            )
            
        epoch_time = time.time() - epoch_start_time
        
        # 计算平均损失和准确率
        train_loss = train_loss / len(self.train_loader) if len(self.train_loader) > 0 else float('inf')
        train_accuracy = 100.0 * train_correct / train_total if train_total > 0 else 0.0
        
        # 记录训练指标
        train_metrics = {
            'loss': train_loss,
            'accuracy': train_accuracy,
            'correct': train_correct,
            'total': train_total,
            'time': epoch_time
        }
        
        return train_metrics 