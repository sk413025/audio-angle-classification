"""
数据追踪模块
用于记录训练过程中的详细样本信息，支持后续分析
"""

import os
import pickle
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import json
import logging

class BatchRecorder:
    """
    批次记录器
    记录训练过程中每个批次的详细信息，包括样本级别的数据
    """
    
    def __init__(self, 
                save_dir, 
                optimizer_type,
                frequency,
                save_full_batch_interval=5, 
                save_samples_with_high_loss=True,
                high_loss_threshold=0.9):
        """
        初始化批次记录器
        
        Args:
            save_dir (str): 保存数据的目录
            optimizer_type (str): 优化器类型 ('svrg' 或 'standard')
            frequency (str): 频率 (如 '500hz')
            save_full_batch_interval (int): 保存完整批次数据的间隔（每隔多少个epoch）
            save_samples_with_high_loss (bool): 是否保存高损失样本
            high_loss_threshold (float): 高损失阈值 (按百分位数，0.9表示损失在前10%)
        """
        self.save_dir = os.path.join(save_dir, optimizer_type, frequency)
        self.batch_dir = os.path.join(self.save_dir, 'batch_records')
        self.sample_dir = os.path.join(self.save_dir, 'sample_records')
        self.metrics_dir = os.path.join(self.save_dir, 'metrics')
        
        os.makedirs(self.batch_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        self.optimizer_type = optimizer_type
        self.frequency = frequency
        self.save_full_batch_interval = save_full_batch_interval
        self.save_samples_with_high_loss = save_samples_with_high_loss
        self.high_loss_threshold = high_loss_threshold
        
        # 用于跟踪每个样本的历史表现
        self.sample_history = {}
        
        # 设置日志
        self.setup_logger()
        
        self.logger.info(f"初始化批次记录器: {optimizer_type} - {frequency}")
        
    def setup_logger(self):
        """设置日志记录器"""
        self.logger = logging.getLogger(f"BatchRecorder_{self.optimizer_type}_{self.frequency}")
        self.logger.setLevel(logging.INFO)
        
        # 创建文件处理器
        log_file = os.path.join(self.save_dir, f"tracker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # 添加处理器到日志记录器
        self.logger.addHandler(file_handler)
    
    def record_batch(self, 
                    epoch, 
                    batch_idx, 
                    data1, 
                    data2, 
                    targets, 
                    label1, 
                    label2, 
                    outputs1, 
                    outputs2, 
                    loss,
                    loss_components=None,
                    gradients=None,
                    model=None):
        """
        记录批次数据
        
        Args:
            epoch (int): 当前epoch
            batch_idx (int): 批次索引
            data1 (torch.Tensor): 第一个样本的数据
            data2 (torch.Tensor): 第二个样本的数据
            targets (torch.Tensor): 目标值
            label1 (list): 第一个样本的标签信息
            label2 (list): 第二个样本的标签信息
            outputs1 (torch.Tensor): 模型对第一个样本的输出
            outputs2 (torch.Tensor): 模型对第二个样本的输出
            loss (torch.Tensor): 批次损失
            loss_components (dict, optional): 损失的各个组成部分
            gradients (dict, optional): 梯度信息
            model (torch.nn.Module, optional): 当前模型，用于提取额外信息
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 计算预测结果
        predictions = (outputs1.detach() > outputs2.detach()) == (targets > 0)
        
        # 获取每个样本的损失值
        if hasattr(loss, 'detach'):
            batch_loss = loss.detach().item()
        else:
            batch_loss = loss
            
        # 创建批次记录
        batch_record = {
            'batch_index': batch_idx,
            'epoch': epoch,
            'timestamp': timestamp,
            'samples': [],
            'batch_stats': {
                'mean_loss': batch_loss,
                'correct_predictions': predictions.sum().item(),
                'total_samples': len(targets),
                'accuracy': predictions.sum().item() / len(targets) if len(targets) > 0 else 0
            }
        }
        
        # 添加梯度信息（如果提供）
        if gradients is not None:
            batch_record['batch_stats']['grad_norm'] = gradients.get('grad_norm', 0)
            batch_record['batch_stats']['svrg_correction_norm'] = gradients.get('svrg_correction_norm', 0)
        
        # 计算每个样本的损失和记录样本信息
        sample_losses = []
        sample_records = []
        
        for i in range(len(targets)):
            # 创建唯一样本ID对（从标签中提取）
            sample1_id = self._create_sample_id(label1[i])
            sample2_id = self._create_sample_id(label2[i])
            pair_id = f"{sample1_id}_{sample2_id}"
            
            # 获取样本输出和预测结果
            output1 = outputs1[i].detach().item()
            output2 = outputs2[i].detach().item()
            target = targets[i].item()
            is_correct = predictions[i].item()
            
            # 创建样本记录
            sample_record = {
                'sample_id': pair_id,
                'metadata': {
                    'sample1': self._extract_metadata(label1[i]),
                    'sample2': self._extract_metadata(label2[i])
                },
                'pair_info': {
                    'sample1_id': sample1_id,
                    'sample2_id': sample2_id,
                    'target': target
                },
                'predictions': {
                    'output1': output1,
                    'output2': output2,
                    'is_correct': bool(is_correct)
                },
                'batch_epoch_info': {
                    'epoch': epoch,
                    'batch_index': batch_idx,
                    'timestamp': timestamp
                }
            }
            
            # 添加到批次记录
            batch_record['samples'].append(sample_record)
            
            # 更新样本历史记录
            if pair_id not in self.sample_history:
                self.sample_history[pair_id] = {
                    'epochs': [],
                    'losses': [],
                    'is_correct': [],
                    'outputs': []
                }
            
            # 更新样本历史
            self.sample_history[pair_id]['epochs'].append(epoch)
            self.sample_history[pair_id]['is_correct'].append(bool(is_correct))
            self.sample_history[pair_id]['outputs'].append((output1, output2))
            
            # 如果需要记录每个样本的损失（如果可用）
            if isinstance(loss, torch.Tensor) and loss.dim() > 0 and loss.size(0) == len(targets):
                sample_loss = loss[i].item()
                sample_record['loss'] = {'value': sample_loss}
                sample_losses.append(sample_loss)
                self.sample_history[pair_id]['losses'].append(sample_loss)
            
            sample_records.append(sample_record)
        
        # 完整批次保存（每隔N个epoch）
        if epoch % self.save_full_batch_interval == 0:
            batch_path = os.path.join(
                self.batch_dir, 
                f"batch_epoch{epoch}_idx{batch_idx}_{timestamp}.pkl"
            )
            with open(batch_path, 'wb') as f:
                pickle.dump(batch_record, f)
            self.logger.info(f"保存完整批次数据: {batch_path}")
        
        # 保存高损失样本
        if self.save_samples_with_high_loss and sample_losses:
            # 计算高损失阈值
            if len(sample_losses) > 1:
                high_loss_threshold = np.quantile(sample_losses, self.high_loss_threshold)
                
                # 筛选高损失样本
                for i, record in enumerate(sample_records):
                    if 'loss' in record and record['loss']['value'] >= high_loss_threshold:
                        # 标记为异常样本
                        record['flags'] = {
                            'is_anomaly': True,
                            'reason': 'high_loss'
                        }
                        
                        # 保存高损失样本记录
                        sample_path = os.path.join(
                            self.sample_dir,
                            f"high_loss_sample_epoch{epoch}_idx{batch_idx}_sample{i}_{timestamp}.pkl"
                        )
                        with open(sample_path, 'wb') as f:
                            pickle.dump(record, f)
                        
                        self.logger.info(f"保存高损失样本: {sample_path}, 损失值: {record['loss']['value']:.4f}")
        
        return batch_record
    
    def _create_sample_id(self, label_info):
        """从标签信息创建唯一样本ID"""
        if isinstance(label_info, list) and len(label_info) >= 3:
            # 假设标签信息包含角度、频率、材质等
            return f"{label_info[0]}_{label_info[1]}_{label_info[2]}"
        elif isinstance(label_info, dict):
            # 如果标签是字典形式
            parts = []
            for key in sorted(label_info.keys()):
                parts.append(str(label_info[key]))
            return "_".join(parts)
        else:
            # 如果无法解析，返回字符串形式
            return str(label_info)
    
    def _extract_metadata(self, label_info):
        """从标签信息提取元数据"""
        if isinstance(label_info, list) and len(label_info) >= 3:
            # 假设标签信息包含[角度, 频率, 材质, ...]
            metadata = {
                'angle': label_info[0] if len(label_info) > 0 else None,
                'frequency': label_info[1] if len(label_info) > 1 else None,
                'material': label_info[2] if len(label_info) > 2 else None
            }
            
            # 添加其他可能的元数据
            if len(label_info) > 3:
                metadata['sequence'] = label_info[3]
            
            return metadata
        elif isinstance(label_info, dict):
            # 如果标签已经是字典形式
            return label_info
        else:
            # 如果无法解析，返回原始值
            return {'raw_label': str(label_info)}
    
    def save_epoch_summary(self, epoch, train_metrics, val_metrics, learning_rate=None):
        """
        保存每个epoch的汇总指标
        
        Args:
            epoch (int): 当前epoch
            train_metrics (dict): 训练指标
            val_metrics (dict): 验证指标
            learning_rate (float, optional): 当前学习率
        """
        # 合并指标
        metrics = {
            'epoch': epoch,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'train': train_metrics,
            'val': val_metrics
        }
        
        if learning_rate is not None:
            metrics['learning_rate'] = learning_rate
        
        # 保存为JSON
        metrics_path = os.path.join(
            self.metrics_dir,
            f"epoch{epoch}_metrics.json"
        )
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"保存Epoch {epoch}指标摘要: {metrics_path}")
        
        # 同时保存为CSV便于后续分析
        metrics_csv_path = os.path.join(self.metrics_dir, "training_metrics.csv")
        
        # 创建扁平化的指标字典用于CSV
        flat_metrics = {
            'epoch': epoch,
            'timestamp': metrics['timestamp']
        }
        
        # 添加训练指标
        for k, v in train_metrics.items():
            flat_metrics[f'train_{k}'] = v
        
        # 添加验证指标
        for k, v in val_metrics.items():
            flat_metrics[f'val_{k}'] = v
        
        if learning_rate is not None:
            flat_metrics['learning_rate'] = learning_rate
        
        # 追加到CSV文件
        df = pd.DataFrame([flat_metrics])
        
        if os.path.exists(metrics_csv_path):
            df.to_csv(metrics_csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(metrics_csv_path, mode='w', header=True, index=False)
    
    def save_sample_histories(self):
        """保存所有样本的历史记录"""
        history_path = os.path.join(
            self.save_dir,
            f"sample_histories_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        )
        with open(history_path, 'wb') as f:
            pickle.dump(self.sample_history, f)
        
        self.logger.info(f"保存样本历史记录: {history_path}, 共{len(self.sample_history)}个样本")
        
        # 同时保存异常样本分析
        self._save_anomaly_analysis()
    
    def _save_anomaly_analysis(self):
        """分析并保存异常样本信息"""
        anomalies = []
        
        for sample_id, history in self.sample_history.items():
            epochs = history['epochs']
            is_correct = history['is_correct']
            
            if len(epochs) < 2:
                continue
            
            # 检查预测不稳定的样本 (频繁变化正确/错误状态)
            changes = sum([1 for i in range(len(is_correct)-1) if is_correct[i] != is_correct[i+1]])
            if changes > len(epochs) // 3:  # 如果变化次数超过总次数的三分之一
                anomalies.append({
                    'sample_id': sample_id,
                    'anomaly_type': 'unstable_predictions',
                    'changes': changes,
                    'total_epochs': len(epochs),
                    'epochs': epochs,
                    'is_correct': is_correct
                })
            
            # 检查损失异常的样本
            if 'losses' in history and len(history['losses']) > 0:
                losses = history['losses']
                avg_loss = sum(losses) / len(losses)
                max_loss = max(losses)
                
                # 如果平均损失特别高
                if avg_loss > 1.5:  # 阈值可以根据实际情况调整
                    anomalies.append({
                        'sample_id': sample_id,
                        'anomaly_type': 'high_average_loss',
                        'avg_loss': avg_loss,
                        'max_loss': max_loss,
                        'epochs': epochs,
                        'losses': losses
                    })
                
                # 如果损失波动很大
                if len(losses) > 2:
                    loss_std = np.std(losses)
                    if loss_std > 0.5:  # 阈值可以根据实际情况调整
                        anomalies.append({
                            'sample_id': sample_id,
                            'anomaly_type': 'unstable_loss',
                            'loss_std': loss_std,
                            'avg_loss': avg_loss,
                            'epochs': epochs,
                            'losses': losses
                        })
        
        # 保存异常样本分析结果
        if anomalies:
            anomaly_path = os.path.join(
                self.save_dir,
                f"anomaly_samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(anomaly_path, 'w') as f:
                json.dump(anomalies, f, indent=2)
            
            self.logger.info(f"保存异常样本分析: {anomaly_path}, 共{len(anomalies)}个异常样本")


class GradientTracker:
    """
    梯度追踪器
    用于记录和分析模型梯度信息
    
    特别适用于SVRG优化器，可记录梯度校正项
    """
    
    def __init__(self, model, save_dir, optimizer_type, frequency):
        """
        初始化梯度追踪器
        
        Args:
            model (torch.nn.Module): 要追踪的模型
            save_dir (str): 保存数据的目录
            optimizer_type (str): 优化器类型 ('svrg' 或 'standard')
            frequency (str): 频率 (如 '500hz')
        """
        self.model = model
        self.save_dir = os.path.join(save_dir, optimizer_type, frequency, 'gradients')
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.optimizer_type = optimizer_type
        self.frequency = frequency
        
        # 设置日志
        self.logger = logging.getLogger(f"GradientTracker_{optimizer_type}_{frequency}")
        self.logger.setLevel(logging.INFO)
        
        # 创建文件处理器
        log_file = os.path.join(self.save_dir, f"grad_tracker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # 添加处理器到日志记录器
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"初始化梯度追踪器: {optimizer_type} - {frequency}")
    
    def compute_gradient_stats(self, normalized=True):
        """
        计算当前模型梯度的统计信息
        
        Args:
            normalized (bool): 是否标准化梯度
            
        Returns:
            dict: 梯度统计信息
        """
        stats = {}
        grad_norm_sum = 0
        param_count = 0
        
        # 统计每个参数的梯度范数
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad).item()
                grad_norm_sum += grad_norm
                param_count += 1
                
                stats[f"grad_norm_{name}"] = grad_norm
                
                # 标准化后的梯度统计
                if normalized and param.data.nelement() > 0:
                    normalized_grad = grad_norm / param.data.nelement()
                    stats[f"norm_grad_{name}"] = normalized_grad
        
        # 平均梯度范数
        if param_count > 0:
            stats["avg_grad_norm"] = grad_norm_sum / param_count
        else:
            stats["avg_grad_norm"] = 0
        
        # 总梯度范数
        stats["total_grad_norm"] = grad_norm_sum
        
        return stats
    
    def compute_svrg_correction(self, snapshot_model, current_model):
        """
        计算SVRG校正项
        
        Args:
            snapshot_model: 快照模型或参数
            current_model: 当前模型或参数
            
        Returns:
            dict: SVRG校正统计信息
        """
        if self.optimizer_type != 'svrg':
            return {}
        
        stats = {}
        correction_norm_sum = 0
        param_count = 0
        
        # 对于SVRG优化器，计算校正项范数
        for name, current_param in current_model.named_parameters():
            if not hasattr(snapshot_model, 'named_parameters'):
                # 如果snapshot_model不是模型而是参数列表
                # 在这种情况下，需要使用索引或其他方式获取对应参数
                continue
                
            for snapshot_name, snapshot_param in snapshot_model.named_parameters():
                if snapshot_name == name and current_param.grad is not None and snapshot_param.grad is not None:
                    # 计算SVRG校正项: current.grad - snapshot.grad + full_grad
                    correction = current_param.grad - snapshot_param.grad
                    correction_norm = torch.norm(correction).item()
                    correction_norm_sum += correction_norm
                    param_count += 1
                    
                    stats[f"correction_norm_{name}"] = correction_norm
                    
                    # 标准化后的校正项
                    if current_param.data.nelement() > 0:
                        normalized_correction = correction_norm / current_param.data.nelement()
                        stats[f"norm_correction_{name}"] = normalized_correction
        
        # 平均校正项范数
        if param_count > 0:
            stats["avg_correction_norm"] = correction_norm_sum / param_count
        else:
            stats["avg_correction_norm"] = 0
        
        # 总校正项范数
        stats["total_correction_norm"] = correction_norm_sum
        
        return stats
    
    def save_gradient_stats(self, epoch, batch_idx, stats):
        """
        保存梯度统计信息
        
        Args:
            epoch (int): 当前epoch
            batch_idx (int): 批次索引
            stats (dict): 梯度统计信息
        """
        # 添加时间戳和索引信息
        stats["epoch"] = epoch
        stats["batch_idx"] = batch_idx
        stats["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存为JSON
        stats_path = os.path.join(
            self.save_dir,
            f"grad_stats_epoch{epoch}_batch{batch_idx}.json"
        )
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"保存梯度统计信息: {stats_path}")
        
        # 同时保存为CSV便于后续分析
        stats_csv_path = os.path.join(self.save_dir, "gradient_stats.csv")
        
        df = pd.DataFrame([stats])
        
        if os.path.exists(stats_csv_path):
            df.to_csv(stats_csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(stats_csv_path, mode='w', header=True, index=False)


class ModelCheckpointer:
    """
    模型检查点保存器
    用于定期保存模型权重和状态
    """
    
    def __init__(self, save_dir, optimizer_type, frequency, save_interval=5):
        """
        初始化模型检查点保存器
        
        Args:
            save_dir (str): 保存数据的目录
            optimizer_type (str): 优化器类型 ('svrg' 或 'standard')
            frequency (str): 频率 (如 '500hz')
            save_interval (int): 保存模型的间隔（每隔多少个epoch）
        """
        self.save_dir = os.path.join(save_dir, optimizer_type, frequency, 'checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.optimizer_type = optimizer_type
        self.frequency = frequency
        self.save_interval = save_interval
        
        # 设置日志
        self.logger = logging.getLogger(f"ModelCheckpointer_{optimizer_type}_{frequency}")
        self.logger.setLevel(logging.INFO)
        
        # 创建文件处理器
        log_file = os.path.join(
            os.path.dirname(self.save_dir), 
            f"checkpointer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # 添加处理器到日志记录器
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"初始化模型检查点保存器: {optimizer_type} - {frequency}")
    
    def save_checkpoint(self, epoch, model, optimizer, scheduler=None, metrics=None, extra_data=None, is_best=False):
        """
        保存模型检查点
        
        Args:
            epoch (int): 当前epoch
            model (torch.nn.Module): 要保存的模型
            optimizer: 优化器
            scheduler: 学习率调度器（可选）
            metrics (dict): 训练指标（可选）
            extra_data (dict): 额外数据（可选）
            is_best (bool): 是否是最佳模型
        """
        # 检查是否应该保存
        should_save = (epoch % self.save_interval == 0) or (epoch == 0) or is_best
        
        if not should_save:
            return
        
        # 准备检查点数据
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_type': self.optimizer_type,
            'frequency': self.frequency,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        # 添加优化器状态
        if optimizer is not None:
            if hasattr(optimizer, 'state_dict'):
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        # 添加调度器状态
        if scheduler is not None:
            if hasattr(scheduler, 'state_dict'):
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # 添加指标
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        # 添加额外数据
        if extra_data is not None:
            checkpoint['extra_data'] = extra_data
        
        # 保存检查点
        filename = f"model_epoch_{epoch}.pt"
        if is_best:
            filename = f"model_best.pt"
        
        checkpoint_path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"保存模型检查点: {checkpoint_path}")
        
        # 如果是最佳模型，也保存一个独立的副本
        if is_best and epoch % self.save_interval != 0:
            best_with_epoch_path = os.path.join(self.save_dir, f"model_best_epoch_{epoch}.pt")
            torch.save(checkpoint, best_with_epoch_path)
            self.logger.info(f"保存最佳模型检查点（带epoch）: {best_with_epoch_path}") 