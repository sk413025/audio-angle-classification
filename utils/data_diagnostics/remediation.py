"""
Data remediation module.

This module provides strategies to address problematic data samples, 
including relabeling suggestions, sample weighting, augmentation strategies,
and synthetic sample generation.
"""

import os
import json
import torch
import numpy as np
import datetime
from torch.utils.data import Dataset, Subset, WeightedRandomSampler, DataLoader, TensorDataset
import copy

class RemediationStrategies:
    """問題數據介入策略，提供改善或移除問題樣本的方法。"""
    
    def __init__(self, detector, dataset, config=None):
        """初始化介入策略器。
        
        Args:
            detector: ProblemSampleDetector 實例，包含已檢測的問題樣本
            dataset: 數據集
            config: 配置參數
        """
        self.detector = detector
        self.dataset = dataset
        self.config = config or {}
        self.log_dir = self.config.get('log_dir', './remediation_logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.strategies = {}
    
    def suggest_relabeling(self, confidence_threshold=0.8, max_samples=100):
        """建議可能需要重新標記的樣本。
        
        Args:
            confidence_threshold: 置信度閾值，低於此值的樣本被考慮
            max_samples: 最多建議樣本數
            
        Returns:
            dict: 包含建議重新標記的樣本及其信息
        """
        print(f"Suggesting samples for relabeling (confidence threshold: {confidence_threshold})...")
        
        # Get problem samples from detector
        problem_indices = self._get_problem_indices()
        
        # If no problem samples, return empty result
        if not problem_indices:
            return {'samples': [], 'confidence_scores': [], 'suggested_labels': []}
        
        # Set model to evaluation mode
        self.detector.model.eval()
        device = self.detector.device
        
        # Store results
        relabel_samples = []
        confidence_scores = []
        suggested_labels = []
        
        # Process in batches for efficiency
        batch_size = 32
        for i in range(0, len(problem_indices), batch_size):
            batch_indices = problem_indices[i:i+batch_size]
            batch_samples = [self.dataset[idx][0] for idx in batch_indices]
            batch_targets = [self.dataset[idx][1] for idx in batch_indices]
            
            # Convert to tensors
            if not isinstance(batch_samples[0], torch.Tensor):
                batch_samples = [torch.tensor(s, dtype=torch.float32) for s in batch_samples]
            
            if not isinstance(batch_targets[0], torch.Tensor):
                batch_targets = [torch.tensor(t) for t in batch_targets]
            
            # Add batch dimension if needed
            batch_samples = [s.unsqueeze(0) if s.dim() == 1 else s for s in batch_samples]
            batch_targets = [t.unsqueeze(0) if t.dim() == 0 else t for t in batch_targets]
            
            # Combine into batches
            inputs = torch.cat(batch_samples, dim=0).to(device)
            targets = torch.cat(batch_targets, dim=0).to(device)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.detector.model(inputs)
                
                # Handle different output shapes
                if outputs.dim() == 1 or outputs.shape[1] == 1:
                    # For regression or ranking tasks
                    predictions = outputs.view(-1).cpu().numpy()
                    confidences = np.ones_like(predictions)  # No confidence for regression, use 1.0
                    if targets.dim() > 1:
                        targets = targets.view(-1)
                else:
                    # For classification tasks
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    max_probs, predictions = torch.max(probabilities, dim=1)
                    confidences = max_probs.cpu().numpy()
                    predictions = predictions.cpu().numpy()
                    if targets.dim() > 1:
                        targets = targets.argmax(dim=1)
                
                targets = targets.cpu().numpy()
            
            # Find samples with low confidence or prediction different from target
            for j, (pred, target, conf, idx) in enumerate(zip(predictions, targets, confidences, batch_indices)):
                if conf < confidence_threshold or pred != target:
                    relabel_samples.append(int(idx))
                    confidence_scores.append(float(conf))
                    suggested_labels.append(int(pred))
        
        # Sort by confidence (lowest first)
        sorted_indices = np.argsort(confidence_scores)
        relabel_samples = [relabel_samples[i] for i in sorted_indices]
        confidence_scores = [confidence_scores[i] for i in sorted_indices]
        suggested_labels = [suggested_labels[i] for i in sorted_indices]
        
        # Limit to max_samples
        if max_samples > 0:
            relabel_samples = relabel_samples[:max_samples]
            confidence_scores = confidence_scores[:max_samples]
            suggested_labels = suggested_labels[:max_samples]
        
        # Store strategy in self.strategies
        strategy_result = {
            'samples': relabel_samples,
            'confidence_scores': confidence_scores,
            'suggested_labels': suggested_labels
        }
        self.strategies['relabeling'] = strategy_result
        
        print(f"Suggested {len(relabel_samples)} samples for relabeling.")
        return strategy_result
    
    def _get_problem_indices(self):
        """Get combined problem sample indices from detector results.
        
        Returns:
            list: Combined problem sample indices
        """
        problem_indices = set()
        
        # Add gradient anomalies
        if 'gradient_anomalies' in self.detector.results:
            problem_indices.update(self.detector.results['gradient_anomalies']['anomaly_indices'])
        
        # Add feature space outliers
        if 'feature_space_outliers' in self.detector.results:
            problem_indices.update(self.detector.results['feature_space_outliers']['outlier_indices'])
        
        # Add consistent errors
        if 'consistent_errors' in self.detector.results:
            problem_indices.update(self.detector.results['consistent_errors']['error_indices'])
        
        return list(problem_indices)
    
    def generate_sample_weights(self, method='inverse_difficulty', alpha=1.0):
        """基於樣本質量生成樣本權重。
        
        Args:
            method: 權重生成方法，支持'inverse_difficulty', 'influence_based'等
            alpha: 權重調整參數
            
        Returns:
            dict 或 numpy.ndarray: 樣本索引到權重的映射
        """
        print(f"Generating sample weights using method: {method} with alpha={alpha}...")
        
        # Get problem samples and their scores
        problem_scores = self._get_problem_scores()
        
        # Create sample weights array
        n_samples = len(self.dataset)
        weights = np.ones(n_samples)
        
        if method == 'inverse_difficulty':
            # Lower weight for more difficult (problematic) samples
            # Get difficulty scores from different detectors
            difficulty_scores = {}
            
            # From gradient anomalies
            if 'gradient_anomalies' in self.detector.results:
                max_grad = max(self.detector.results['gradient_anomalies']['gradient_values']) if self.detector.results['gradient_anomalies']['gradient_values'] else 1.0
                for idx, grad in zip(self.detector.results['gradient_anomalies']['anomaly_indices'],
                                   self.detector.results['gradient_anomalies']['gradient_values']):
                    difficulty_scores[idx] = difficulty_scores.get(idx, 0) + grad / max_grad
            
            # From feature space outliers
            if 'feature_space_outliers' in self.detector.results:
                max_score = max(self.detector.results['feature_space_outliers']['outlier_scores']) if self.detector.results['feature_space_outliers']['outlier_scores'] else 1.0
                for idx, score in zip(self.detector.results['feature_space_outliers']['outlier_indices'],
                                    self.detector.results['feature_space_outliers']['outlier_scores']):
                    difficulty_scores[idx] = difficulty_scores.get(idx, 0) + score / max_score
            
            # From consistent errors
            if 'consistent_errors' in self.detector.results:
                for idx, rate in zip(self.detector.results['consistent_errors']['error_indices'],
                                   self.detector.results['consistent_errors']['error_rates']):
                    difficulty_scores[idx] = difficulty_scores.get(idx, 0) + rate
            
            # Normalize combined difficulty scores and apply inverse weighting
            for idx, score in difficulty_scores.items():
                # Higher difficulty score -> lower weight
                weights[idx] = 1.0 / (1.0 + alpha * score)
            
        elif method == 'influence_based':
            # Requires pre-computed influence scores
            # This is a simplified version; actual influence calculation is more complex
            
            # Use consistent errors as a proxy for influence
            if 'consistent_errors' in self.detector.results:
                for idx, rate in zip(self.detector.results['consistent_errors']['error_indices'],
                                   self.detector.results['consistent_errors']['error_rates']):
                    # Higher error rate -> higher influence -> lower weight
                    weights[idx] = 1.0 / (1.0 + alpha * rate)
            
        elif method == 'class_balanced':
            # Balance weights based on class distribution
            # Get all labels
            labels = []
            for i in range(n_samples):
                _, label = self.dataset[i]
                if isinstance(label, torch.Tensor):
                    if label.dim() > 0 and label.size(0) > 1:
                        # One-hot encoded
                        label = label.argmax().item()
                    else:
                        label = label.item()
                labels.append(label)
            
            # Count instances per class
            class_counts = {}
            for label in labels:
                class_counts[label] = class_counts.get(label, 0) + 1
            
            # Calculate inverse frequency weights
            for i, label in enumerate(labels):
                weights[i] = n_samples / (len(class_counts) * class_counts[label])
            
            # Apply additional scaling for problem samples
            problem_indices = self._get_problem_indices()
            for idx in problem_indices:
                if idx < n_samples:
                    # Give problematic samples even lower weight
                    weights[idx] = weights[idx] / (1.0 + alpha)
            
        else:
            raise ValueError(f"Unsupported weight generation method: {method}")
        
        # Normalize weights to have mean of 1.0
        if np.sum(weights) > 0:
            weights = weights * (n_samples / np.sum(weights))
        
        # Create result dictionary
        weights_dict = {i: float(w) for i, w in enumerate(weights)}
        
        # Store strategy in self.strategies
        self.strategies['sample_weights'] = {
            'method': method,
            'alpha': alpha,
            'weights': weights_dict
        }
        
        return weights_dict
    
    def suggest_augmentation_strategies(self, problem_samples=None, strategy_pool=None):
        """針對問題樣本建議數據增強策略。
        
        Args:
            problem_samples: 問題樣本索引，None則使用detector識別的樣本
            strategy_pool: 可用增強策略集合
            
        Returns:
            dict: 樣本到建議增強策略的映射
        """
        print("Suggesting augmentation strategies for problem samples...")
        
        # Default strategy pool if not provided
        if strategy_pool is None:
            strategy_pool = [
                'random_noise', 
                'gaussian_blur', 
                'rotation', 
                'scaling',
                'flip',
                'time_stretching'  # For audio/time series data
            ]
        
        # Get problem samples if not provided
        if problem_samples is None:
            problem_samples = self._get_problem_indices()
        
        # If no problem samples, return empty result
        if not problem_samples:
            return {}
        
        # Get problem types for each sample
        problem_types = self._get_problem_types(problem_samples)
        
        # Define strategy rules based on problem types
        augmentation_strategies = {}
        
        for idx in problem_samples:
            # Get problem types for this sample
            types = problem_types.get(idx, [])
            suggested_strategies = []
            
            if 'gradient_anomaly' in types:
                # For gradient anomalies, try noise and blur to reduce sharp gradients
                if 'random_noise' in strategy_pool:
                    suggested_strategies.append('random_noise')
                if 'gaussian_blur' in strategy_pool:
                    suggested_strategies.append('gaussian_blur')
            
            if 'feature_outlier' in types:
                # For feature outliers, try transformations that might bring it closer to the data manifold
                if 'rotation' in strategy_pool:
                    suggested_strategies.append('rotation')
                if 'scaling' in strategy_pool:
                    suggested_strategies.append('scaling')
            
            if 'consistent_error' in types:
                # For consistent errors, try more aggressive augmentations
                if 'flip' in strategy_pool:
                    suggested_strategies.append('flip')
                if 'time_stretching' in strategy_pool:
                    suggested_strategies.append('time_stretching')
            
            # If no specific strategies, suggest random ones
            if not suggested_strategies and strategy_pool:
                # Select a random subset of strategies
                n_strategies = min(2, len(strategy_pool))
                suggested_strategies = list(np.random.choice(strategy_pool, n_strategies, replace=False))
            
            augmentation_strategies[idx] = suggested_strategies
        
        # Store strategy in self.strategies
        self.strategies['augmentation'] = {
            'problem_samples': problem_samples,
            'augmentation_strategies': augmentation_strategies
        }
        
        return augmentation_strategies
    
    def _get_problem_scores(self):
        """Compute problem scores for each sample based on detector results.
        
        Returns:
            dict: Sample indices to problem scores mapping
        """
        problem_scores = {}
        
        # Add gradient anomaly scores
        if 'gradient_anomalies' in self.detector.results:
            for idx, grad in zip(self.detector.results['gradient_anomalies']['anomaly_indices'],
                               self.detector.results['gradient_anomalies']['gradient_values']):
                if idx not in problem_scores:
                    problem_scores[idx] = {}
                problem_scores[idx]['gradient'] = grad
        
        # Add outlier scores
        if 'feature_space_outliers' in self.detector.results:
            for idx, score in zip(self.detector.results['feature_space_outliers']['outlier_indices'],
                                self.detector.results['feature_space_outliers']['outlier_scores']):
                if idx not in problem_scores:
                    problem_scores[idx] = {}
                problem_scores[idx]['outlier'] = score
        
        # Add error rates
        if 'consistent_errors' in self.detector.results:
            for idx, rate in zip(self.detector.results['consistent_errors']['error_indices'],
                               self.detector.results['consistent_errors']['error_rates']):
                if idx not in problem_scores:
                    problem_scores[idx] = {}
                problem_scores[idx]['error_rate'] = rate
        
        return problem_scores
    
    def _get_problem_types(self, indices):
        """Determine problem types for each sample index.
        
        Args:
            indices: List of sample indices
            
        Returns:
            dict: Mapping from sample index to list of problem types
        """
        problem_types = {}
        
        for idx in indices:
            problem_types[idx] = []
            
            # Check for gradient anomalies
            if 'gradient_anomalies' in self.detector.results:
                if idx in self.detector.results['gradient_anomalies']['anomaly_indices']:
                    problem_types[idx].append('gradient_anomaly')
            
            # Check for feature space outliers
            if 'feature_space_outliers' in self.detector.results:
                if idx in self.detector.results['feature_space_outliers']['outlier_indices']:
                    problem_types[idx].append('feature_outlier')
            
            # Check for consistent errors
            if 'consistent_errors' in self.detector.results:
                if idx in self.detector.results['consistent_errors']['error_indices']:
                    problem_types[idx].append('consistent_error')
        
        return problem_types
    
    def generate_synthetic_samples(self, method='smote', problem_class_indices=None, n_samples=None):
        """為問題類別生成合成樣本。
        
        Args:
            method: 合成樣本生成方法
            problem_class_indices: 問題類別索引，None則自動識別
            n_samples: 每個類別生成樣本數
            
        Returns:
            tuple: (合成樣本, 標籤)
        """
        print(f"Generating synthetic samples using method: {method}...")
        
        # Get all data and labels
        all_data = []
        all_labels = []
        for i in range(len(self.dataset)):
            data, label = self.dataset[i]
            
            # Ensure data is a tensor
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=torch.float32)
            
            # Ensure label is a tensor
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label)
            
            all_data.append(data.view(1, -1))
            if label.dim() > 0 and label.size(0) > 1:  # One-hot encoded
                all_labels.append(label.argmax().view(1))
            else:
                all_labels.append(label.view(1))
        
        # Concatenate into tensors
        all_data = torch.cat(all_data, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Auto-identify problem classes if not provided
        if problem_class_indices is None:
            problem_indices = self._get_problem_indices()
            problem_labels = all_labels[problem_indices].unique().tolist()
        else:
            problem_labels = problem_class_indices
            
        # Default number of samples to generate
        if n_samples is None:
            n_samples = max(10, len(problem_indices) // 2)
        
        # Initialize synthetic data and labels
        synthetic_data = []
        synthetic_labels = []
        
        # Generate synthetic samples for each problem class
        for label in problem_labels:
            # Get samples of this class
            class_indices = (all_labels == label).nonzero().view(-1)
            class_data = all_data[class_indices]
            
            # Skip if too few samples
            if len(class_indices) < 2:
                continue
                
            if method == 'smote':
                # Simplified SMOTE implementation
                for _ in range(n_samples):
                    # Select a random sample
                    idx1 = torch.randint(0, len(class_indices), (1,)).item()
                    # Select a different random sample
                    idx2 = idx1
                    while idx2 == idx1:
                        idx2 = torch.randint(0, len(class_indices), (1,)).item()
                    
                    # Generate a synthetic sample by interpolation
                    alpha = torch.rand(1).item()
                    synthetic_sample = class_data[idx1] * alpha + class_data[idx2] * (1 - alpha)
                    
                    synthetic_data.append(synthetic_sample.view(1, -1))
                    synthetic_labels.append(torch.tensor([label], dtype=all_labels.dtype))
            
            elif method == 'random_noise':
                # Add random noise to existing samples
                for _ in range(n_samples):
                    idx = torch.randint(0, len(class_indices), (1,)).item()
                    noise = torch.randn_like(class_data[idx]) * 0.1
                    synthetic_sample = class_data[idx] + noise
                    
                    synthetic_data.append(synthetic_sample.view(1, -1))
                    synthetic_labels.append(torch.tensor([label], dtype=all_labels.dtype))
            
            elif method == 'average':
                # Create new samples by averaging existing ones
                for _ in range(n_samples):
                    # Select multiple random samples
                    k = min(5, len(class_indices))
                    indices = torch.randperm(len(class_indices))[:k]
                    # Average them
                    synthetic_sample = class_data[indices].mean(dim=0)
                    
                    synthetic_data.append(synthetic_sample.view(1, -1))
                    synthetic_labels.append(torch.tensor([label], dtype=all_labels.dtype))
            
            else:
                raise ValueError(f"Unsupported synthetic sample generation method: {method}")
        
        # Combine all synthetic samples
        if not synthetic_data:
            # Return empty tensors if no synthetic data
            return torch.tensor([]), torch.tensor([])
            
        synthetic_data = torch.cat(synthetic_data, dim=0)
        synthetic_labels = torch.cat(synthetic_labels, dim=0)
        
        # Store strategy in self.strategies
        self.strategies['synthetic_samples'] = {
            'method': method,
            'problem_class_indices': problem_labels,
            'n_samples': n_samples,
            'n_generated': len(synthetic_data)
        }
        
        print(f"Generated {len(synthetic_data)} synthetic samples for classes {problem_labels}.")
        return synthetic_data, synthetic_labels
    
    def apply_remediation(self, dataset, strategy, **kwargs):
        """應用選定的介入策略到數據集。
        
        Args:
            dataset: 要應用策略的數據集
            strategy: 策略名稱或策略配置
            **kwargs: 策略參數
            
        Returns:
            object: 應用策略後的數據集
        """
        print(f"Applying remediation strategy: {strategy}...")
        
        # Create a copy of the dataset to avoid modifying the original
        remediated_dataset = None
        
        if strategy == "weighted_sampling":
            # Use weighted sampling during data loading
            weights = kwargs.get('weights', None)
            if weights is None:
                weights = self.generate_sample_weights()
            
            # Convert weights to a list if it's a dictionary
            if isinstance(weights, dict):
                weight_list = [weights.get(i, 1.0) for i in range(len(dataset))]
            else:
                weight_list = weights
            
            # Create a weighted sampler
            sampler = WeightedRandomSampler(
                weights=weight_list,
                num_samples=len(dataset),
                replacement=True
            )
            
            # Return the original dataset with a sampler
            # This is a bit hacky but allows us to return something that works with DataLoader
            class WeightedDataset(Dataset):
                def __init__(self, dataset, sampler):
                    self.dataset = dataset
                    self.sampler = sampler
                
                def __len__(self):
                    return len(self.dataset)
                
                def __getitem__(self, idx):
                    return self.dataset[idx]
                
                def get_sampler(self):
                    return self.sampler
            
            remediated_dataset = WeightedDataset(dataset, sampler)
            
        elif strategy == "remove_samples":
            # Remove problematic samples
            indices_to_remove = kwargs.get('indices_to_remove', None)
            if indices_to_remove is None:
                indices_to_remove = self._get_problem_indices()
            
            # Create a subset without the problematic samples
            keep_indices = [i for i in range(len(dataset)) if i not in indices_to_remove]
            remediated_dataset = Subset(dataset, keep_indices)
            
        elif strategy == "augmentation":
            # Apply data augmentation
            augmentation_strategies = kwargs.get('augmentation_strategies', None)
            if augmentation_strategies is None:
                augmentation_strategies = self.suggest_augmentation_strategies()
            
            # Create an augmented dataset
            class AugmentedDataset(Dataset):
                def __init__(self, dataset, augmentation_strategies):
                    self.dataset = dataset
                    self.augmentation_strategies = augmentation_strategies
                    
                    # Define augmentation functions
                    self.augmentations = {
                        'random_noise': lambda x: x + torch.randn_like(x) * 0.1,
                        'gaussian_blur': lambda x: x,  # Placeholder, implement real blur
                        'rotation': lambda x: x,  # Placeholder, implement rotation
                        'scaling': lambda x: x * torch.rand(1).item() * 0.5 + 0.75,
                        'flip': lambda x: x.flip(dims=[-1]),
                        'time_stretching': lambda x: x  # Placeholder
                    }
                
                def __len__(self):
                    return len(self.dataset)
                
                def __getitem__(self, idx):
                    x, y = self.dataset[idx]
                    
                    # Apply augmentations if this is a problem sample
                    if idx in self.augmentation_strategies:
                        for aug_name in self.augmentation_strategies[idx]:
                            if aug_name in self.augmentations:
                                aug_func = self.augmentations[aug_name]
                                if torch.rand(1).item() < 0.5:  # 50% chance to apply
                                    x = aug_func(x)
                    
                    return x, y
            
            remediated_dataset = AugmentedDataset(dataset, augmentation_strategies)
            
        elif strategy == "synthetic_samples":
            # Add synthetic samples to the dataset
            synthetic_data = kwargs.get('synthetic_data', None)
            synthetic_labels = kwargs.get('synthetic_labels', None)
            
            if synthetic_data is None or synthetic_labels is None:
                synthetic_data, synthetic_labels = self.generate_synthetic_samples()
            
            if len(synthetic_data) == 0:
                # No synthetic samples, return original dataset
                remediated_dataset = dataset
            else:
                # Create a new dataset with original and synthetic samples
                class EnhancedDataset(Dataset):
                    def __init__(self, original_dataset, synthetic_data, synthetic_labels):
                        self.original_dataset = original_dataset
                        self.synthetic_data = synthetic_data
                        self.synthetic_labels = synthetic_labels
                        self.original_len = len(original_dataset)
                        self.synthetic_len = len(synthetic_data)
                    
                    def __len__(self):
                        return self.original_len + self.synthetic_len
                    
                    def __getitem__(self, idx):
                        if idx < self.original_len:
                            return self.original_dataset[idx]
                        else:
                            # Return synthetic sample
                            syn_idx = idx - self.original_len
                            return self.synthetic_data[syn_idx], self.synthetic_labels[syn_idx]
                
                remediated_dataset = EnhancedDataset(dataset, synthetic_data, synthetic_labels)
                
        else:
            raise ValueError(f"Unsupported remediation strategy: {strategy}")
        
        # Store strategy in self.strategies
        self.strategies['applied_remediation'] = {
            'strategy': strategy,
            'parameters': kwargs,
            'dataset_size': len(remediated_dataset) if remediated_dataset is not None else 0
        }
        
        return remediated_dataset
    
    def evaluate_remediation_effect(self, original_model, remediated_dataset, eval_metric='accuracy'):
        """評估介入策略的效果。
        
        Args:
            original_model: 原始模型
            remediated_dataset: 應用介入策略後的數據集
            eval_metric: 評估指標
            
        Returns:
            dict: 包含評估結果的字典
        """
        print(f"Evaluating remediation effect using metric: {eval_metric}...")
        
        # Put model in evaluation mode
        original_model.eval()
        device = next(original_model.parameters()).device
        
        # Function to compute accuracy
        def compute_accuracy(model, dataset, device):
            correct = 0
            total = 0
            
            # Use DataLoader to batch the data
            loader = DataLoader(dataset, batch_size=32, shuffle=False)
            
            with torch.no_grad():
                for inputs, targets in loader:
                    inputs = inputs.to(device)
                    
                    # Handle different target formats
                    if isinstance(targets, torch.Tensor):
                        if targets.dim() > 1 and targets.size(1) > 1:
                            # One-hot encoded
                            targets = targets.argmax(dim=1)
                        targets = targets.to(device)
                    
                    # Get predictions
                    outputs = model(inputs)
                    
                    # Handle different output formats
                    if outputs.dim() == 1 or outputs.size(1) == 1:
                        # Regression or binary classification
                        predictions = (outputs > 0.5).long().view(-1)
                    else:
                        # Multi-class classification
                        _, predictions = torch.max(outputs, 1)
                    
                    # Update counts
                    total += targets.size(0)
                    correct += (predictions == targets).sum().item()
            
            return correct / total if total > 0 else 0.0
        
        # Compute accuracy on original dataset
        original_accuracy = compute_accuracy(original_model, self.dataset, device)
        
        # Compute accuracy on remediated dataset
        remediated_accuracy = compute_accuracy(original_model, remediated_dataset, device)
        
        # Compute difference
        improvement = remediated_accuracy - original_accuracy
        
        # Create results dictionary
        results = {
            'metric': eval_metric,
            'original_metrics': {eval_metric: original_accuracy},
            'remediated_metrics': {eval_metric: remediated_accuracy},
            'improvement': improvement,
            'relative_improvement': improvement / original_accuracy if original_accuracy > 0 else float('inf')
        }
        
        # Store results in self.strategies
        self.strategies['evaluation'] = results
        
        print(f"Original {eval_metric}: {original_accuracy:.4f}")
        print(f"Remediated {eval_metric}: {remediated_accuracy:.4f}")
        print(f"Improvement: {improvement:.4f} ({results['relative_improvement']:.2%})")
        
        return results
    
    def log_remediation_process(self, strategy, params, results):
        """記錄介入過程和結果。
        
        Args:
            strategy: 使用的策略
            params: 策略參數
            results: 介入結果
            
        Returns:
            str: 日誌文件路徑
        """
        print(f"Logging remediation process for strategy: {strategy}...")
        
        # Create timestamp for filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"remediation_{strategy}_{timestamp}.json"
        filepath = os.path.join(self.log_dir, filename)
        
        # Create log data
        log_data = {
            'timestamp': timestamp,
            'strategy': strategy,
            'parameters': params,
            'results': results,
            'detector_info': {
                'n_gradient_anomalies': len(self.detector.results.get('gradient_anomalies', {}).get('anomaly_indices', [])),
                'n_feature_outliers': len(self.detector.results.get('feature_space_outliers', {}).get('outlier_indices', [])),
                'n_consistent_errors': len(self.detector.results.get('consistent_errors', {}).get('error_indices', []))
            }
        }
        
        # Save log to file
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"Remediation process logged to {filepath}")
        return filepath 