"""
增强优化器模块
为标准优化器和SVRG优化器添加增强功能
"""

import torch
from torch.optim import Optimizer, Adam, SGD
import copy
import numpy as np
from utils.svrg_optimizer import SVRG_k, SVRG_Snapshot

class TrackedAdam(Adam):
    """
    增强的Adam优化器，添加了梯度追踪功能
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, name="Adam"):
        super(TrackedAdam, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)
        self.name = name
        self.grad_history = []
        self.step_count = 0
    
    def step(self, closure=None, compute_stats=False, record_grad=False):
        """
        执行优化步骤，可选记录梯度统计
        
        Args:
            closure (callable, optional): 返回损失的闭包函数
            compute_stats (bool): 是否计算梯度统计信息
            record_grad (bool): 是否记录梯度到历史记录
            
        Returns:
            loss or stats: 如果compute_stats为True，返回梯度统计信息
        """
        self.step_count += 1
        
        if compute_stats:
            stats = self._compute_grad_stats()
            
        loss = super(TrackedAdam, self).step(closure)
        
        if record_grad:
            self._record_grad_to_history()
            
        if compute_stats:
            return stats
        
        return loss
    
    def _compute_grad_stats(self):
        """计算当前梯度的统计信息"""
        stats = {}
        grad_norm_sum = 0
        max_grad_norm = 0
        min_grad_norm = float('inf')
        grad_norms = []
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad_norm = torch.norm(p.grad).item()
                    grad_norm_sum += grad_norm
                    max_grad_norm = max(max_grad_norm, grad_norm)
                    min_grad_norm = min(min_grad_norm, grad_norm)
                    grad_norms.append(grad_norm)
        
        if grad_norms:
            stats['grad_norm'] = grad_norm_sum
            stats['avg_grad_norm'] = grad_norm_sum / len(grad_norms)
            stats['max_grad_norm'] = max_grad_norm
            stats['min_grad_norm'] = min_grad_norm
            stats['std_grad_norm'] = np.std(grad_norms) if len(grad_norms) > 1 else 0
        else:
            stats['grad_norm'] = 0
            stats['avg_grad_norm'] = 0
            stats['max_grad_norm'] = 0
            stats['min_grad_norm'] = 0
            stats['std_grad_norm'] = 0
            
        stats['step_count'] = self.step_count
        
        return stats
    
    def _record_grad_to_history(self):
        """记录当前梯度到历史记录"""
        grad_snapshot = []
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad_snapshot.append(p.grad.clone().detach())
        
        self.grad_history.append({
            'step': self.step_count,
            'gradients': grad_snapshot
        })
    
    def get_lr(self):
        """获取当前学习率"""
        for group in self.param_groups:
            return group['lr']
        return None


class TrackedSGD(SGD):
    """
    增强的SGD优化器，添加了梯度追踪功能和动量支持
    """
    
    def __init__(self, params, lr=0.01, momentum=0.9, dampening=0, 
                 weight_decay=0, nesterov=False, name="SGD_Momentum"):
        super(TrackedSGD, self).__init__(
            params, lr, momentum, dampening, weight_decay, nesterov
        )
        self.name = name
        self.grad_history = []
        self.step_count = 0
        self.momentum = momentum
        self.nesterov = nesterov
    
    def step(self, closure=None, compute_stats=False, record_grad=False):
        """
        执行优化步骤，可选记录梯度统计
        
        Args:
            closure (callable, optional): 返回损失的闭包函数
            compute_stats (bool): 是否计算梯度统计信息
            record_grad (bool): 是否记录梯度到历史记录
            
        Returns:
            loss or stats: 如果compute_stats为True，返回梯度统计信息
        """
        self.step_count += 1
        
        if compute_stats:
            stats = self._compute_grad_stats()
            
        loss = super(TrackedSGD, self).step(closure)
        
        if record_grad:
            self._record_grad_to_history()
            
        if compute_stats:
            return stats
        
        return loss
    
    def _compute_grad_stats(self):
        """计算当前梯度的统计信息"""
        stats = {}
        grad_norm_sum = 0
        max_grad_norm = 0
        min_grad_norm = float('inf')
        grad_norms = []
        
        # 计算动量项的统计信息
        momentum_norm_sum = 0
        momentum_norms = []
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad_norm = torch.norm(p.grad).item()
                    grad_norm_sum += grad_norm
                    max_grad_norm = max(max_grad_norm, grad_norm)
                    min_grad_norm = min(min_grad_norm, grad_norm)
                    grad_norms.append(grad_norm)
                    
                    # 如果存在momentum缓冲区，计算动量统计信息
                    param_state = self.state[p]
                    if 'momentum_buffer' in param_state:
                        momentum_buffer = param_state['momentum_buffer']
                        momentum_norm = torch.norm(momentum_buffer).item()
                        momentum_norm_sum += momentum_norm
                        momentum_norms.append(momentum_norm)
        
        if grad_norms:
            stats['grad_norm'] = grad_norm_sum
            stats['avg_grad_norm'] = grad_norm_sum / len(grad_norms)
            stats['max_grad_norm'] = max_grad_norm
            stats['min_grad_norm'] = min_grad_norm
            stats['std_grad_norm'] = np.std(grad_norms) if len(grad_norms) > 1 else 0
        else:
            stats['grad_norm'] = 0
            stats['avg_grad_norm'] = 0
            stats['max_grad_norm'] = 0
            stats['min_grad_norm'] = 0
            stats['std_grad_norm'] = 0
            
        # 添加动量相关统计信息
        if momentum_norms:
            stats['momentum_norm'] = momentum_norm_sum
            stats['avg_momentum_norm'] = momentum_norm_sum / len(momentum_norms)
            stats['momentum_factor'] = self.momentum
            stats['nesterov'] = self.nesterov
        
        stats['step_count'] = self.step_count
        
        return stats
    
    def _record_grad_to_history(self):
        """记录当前梯度到历史记录"""
        grad_snapshot = []
        momentum_snapshot = []
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad_snapshot.append(p.grad.clone().detach())
                    
                    # 记录动量缓冲区
                    param_state = self.state[p]
                    if 'momentum_buffer' in param_state:
                        momentum_snapshot.append(param_state['momentum_buffer'].clone().detach())
        
        self.grad_history.append({
            'step': self.step_count,
            'gradients': grad_snapshot,
            'momentum_buffers': momentum_snapshot if momentum_snapshot else None
        })
    
    def get_lr(self):
        """获取当前学习率"""
        for group in self.param_groups:
            return group['lr']
        return None


class TrackedSVRG(SVRG_k):
    """
    增强的SVRG优化器，添加了梯度追踪功能
    """
    
    def __init__(self, params, lr=1e-3, weight_decay=0, name="SVRG"):
        super(TrackedSVRG, self).__init__(params, lr, weight_decay)
        self.name = name
        self.grad_history = []
        self.correction_history = []
        self.step_count = 0
    
    def step(self, params, compute_stats=False, record_grad=False):
        """
        执行SVRG优化步骤，可选记录梯度统计
        
        Args:
            params: 快照参数
            compute_stats (bool): 是否计算梯度统计信息
            record_grad (bool): 是否记录梯度到历史记录
            
        Returns:
            stats: 如果compute_stats为True，返回梯度统计信息
        """
        self.step_count += 1
        stats = None
        
        if compute_stats:
            stats = self._compute_grad_stats(params)
        
        super(TrackedSVRG, self).step(params)
        
        if record_grad:
            self._record_grad_to_history(params)
        
        return stats if compute_stats else None
    
    def _compute_grad_stats(self, params):
        """计算SVRG梯度统计，包括校正项"""
        stats = {}
        grad_norm_sum = 0
        correction_norm_sum = 0
        max_correction_norm = 0
        correction_norms = []
        
        for group, new_group, u_group in zip(self.param_groups, params, self.u):
            for p, q, u in zip(group['params'], new_group['params'], u_group['params']):
                if p.grad is None or q.grad is None or u.grad is None:
                    continue
                
                # 常规梯度范数
                grad_norm = torch.norm(p.grad).item()
                grad_norm_sum += grad_norm
                
                # SVRG校正项: p.grad - q.grad + u.grad
                correction = p.grad.data - q.grad.data + u.grad.data
                correction_norm = torch.norm(correction).item()
                correction_norm_sum += correction_norm
                max_correction_norm = max(max_correction_norm, correction_norm)
                correction_norms.append(correction_norm)
        
        stats['grad_norm'] = grad_norm_sum
        stats['correction_norm'] = correction_norm_sum
        if correction_norms:
            stats['avg_correction_norm'] = correction_norm_sum / len(correction_norms)
            stats['max_correction_norm'] = max_correction_norm
            stats['std_correction_norm'] = np.std(correction_norms) if len(correction_norms) > 1 else 0
        else:
            stats['avg_correction_norm'] = 0
            stats['max_correction_norm'] = 0
            stats['std_correction_norm'] = 0
            
        stats['step_count'] = self.step_count
        
        return stats
    
    def _record_grad_to_history(self, params):
        """记录当前梯度和校正项到历史记录"""
        grad_snapshot = []
        correction_snapshot = []
        
        for group, new_group, u_group in zip(self.param_groups, params, self.u):
            for p, q, u in zip(group['params'], new_group['params'], u_group['params']):
                if p.grad is None or q.grad is None or u.grad is None:
                    continue
                
                grad_snapshot.append(p.grad.clone().detach())
                
                # 记录SVRG校正项
                correction = p.grad.data - q.grad.data + u.grad.data
                correction_snapshot.append(correction.clone().detach())
        
        self.grad_history.append({
            'step': self.step_count,
            'gradients': grad_snapshot
        })
        
        self.correction_history.append({
            'step': self.step_count,
            'corrections': correction_snapshot
        })
    
    def get_lr(self):
        """获取当前学习率"""
        for group in self.param_groups:
            return group['lr']
        return None


class TrackedSVRGSnapshot(SVRG_Snapshot):
    """
    增强的SVRG快照，添加了统计信息
    """
    
    def __init__(self, params, name="SVRG_Snapshot"):
        super(TrackedSVRGSnapshot, self).__init__(params)
        self.name = name
        self.update_count = 0
    
    def set_param_groups(self, new_params):
        """复制参数并记录更新次数"""
        super(TrackedSVRGSnapshot, self).set_param_groups(new_params)
        self.update_count += 1
        return self.update_count 