"""
Problem sample detection module.

This module provides functionality to identify problematic samples in a dataset
using various detection methods, including gradient anomaly detection, consistent
error detection, and feature space outlier detection.
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN

class ProblemSampleDetector:
    """問題樣本識別器，提供多種方法來識別數據集中的問題樣本。"""
    
    def __init__(self, model, dataset, config=None):
        """初始化問題樣本識別器。
        
        Args:
            model: 已訓練的模型
            dataset: 要分析的數據集
            config: 配置參數
        """
        self.model = model
        self.dataset = dataset
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                  "mps" if torch.backends.mps.is_available() else
                                  "cpu")
        self.results = {}
    
    def detect_gradient_anomalies(self, criterion=None, threshold=0.95):
        """使用梯度信息識別異常樣本。
        
        Args:
            criterion: 損失函數，默認使用交叉熵損失
            threshold: 異常判定閾值，梯度大於總體分布多少百分位
            
        Returns:
            dict: 包含異常樣本索引和其相應梯度值
        """
        print(f"Detecting gradient anomalies with threshold {threshold}...")
        
        # Default to cross-entropy loss if no criterion provided
        if criterion is None:
            criterion = torch.nn.CrossEntropyLoss(reduction='none')
        
        # Set model to evaluation mode
        self.model.eval()
        self.model.to(self.device)
        
        # Store gradients for each sample
        gradients = []
        all_indices = []
        
        # Use DataLoader to process in batches
        batch_size = self.config.get('batch_size', 32)
        dataloader = DataLoader(
            self.dataset, 
            batch_size=batch_size,
            shuffle=False, 
            num_workers=0
        )
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            batch_indices = list(range(batch_idx * batch_size, min((batch_idx + 1) * batch_size, len(self.dataset))))
            all_indices.extend(batch_indices)
            
            # Move data to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Enable gradient tracking
            inputs.requires_grad = True
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Handle different output shapes based on the model
            if outputs.dim() == 1 or outputs.shape[1] == 1:
                # If the model outputs a single score (e.g., for ranking or regression)
                outputs = outputs.view(-1, 1)
                if targets.dim() == 1:
                    targets = targets.float().view(-1, 1)
                loss = torch.nn.functional.mse_loss(outputs, targets.float(), reduction='none')
            else:
                # For classification with multiple classes
                if targets.dim() == 2:
                    targets = targets.argmax(dim=1)
                loss = criterion(outputs, targets)
            
            # Get gradients
            for i in range(len(inputs)):
                # Zero gradients
                if inputs.grad is not None:
                    inputs.grad.zero_()
                
                # Backward pass for individual sample
                if loss.dim() > 0:
                    sample_loss = loss[i].mean()
                else:
                    sample_loss = loss
                sample_loss.backward(retain_graph=True)
                
                # Compute gradient magnitude
                if inputs.grad is not None:
                    grad_magnitude = inputs.grad[i].abs().mean().item()
                    gradients.append(grad_magnitude)
            
            # Free memory
            inputs.requires_grad = False
            inputs.grad = None
        
        # Find anomalies based on threshold
        gradients = np.array(gradients)
        threshold_value = np.percentile(gradients, threshold * 100)
        anomaly_indices = [idx for idx, grad in zip(all_indices, gradients) if grad > threshold_value]
        anomaly_gradients = [grad for idx, grad in zip(all_indices, gradients) if grad > threshold_value]
        
        # Store results
        result = {
            'anomaly_indices': anomaly_indices,
            'gradient_values': anomaly_gradients,
            'threshold': threshold_value,
            'percentile': threshold
        }
        self.results['gradient_anomalies'] = result
        print(f"Found {len(anomaly_indices)} gradient anomalies out of {len(gradients)} samples.")
        
        return result
    
    def detect_consistent_errors(self, n_folds=5, seed=42):
        """識別模型在交叉驗證中一致預測錯誤的樣本。
        
        Args:
            n_folds: 交叉驗證折數
            seed: 隨機種子
            
        Returns:
            dict: 包含一致錯誤的樣本索引及其錯誤率
        """
        print(f"Detecting consistent errors with {n_folds}-fold cross-validation...")
        
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize error counters for each sample
        n_samples = len(self.dataset)
        error_counts = np.zeros(n_samples)
        
        # Create indices for all samples
        all_indices = list(range(n_samples))
        
        # Create folds
        fold_size = n_samples // n_folds
        folds = []
        for i in range(n_folds):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < n_folds - 1 else n_samples
            test_indices = all_indices[start_idx:end_idx]
            train_indices = [idx for idx in all_indices if idx not in test_indices]
            folds.append((train_indices, test_indices))
        
        # Perform cross-validation
        for fold_idx, (train_indices, test_indices) in enumerate(folds):
            print(f"Processing fold {fold_idx+1}/{n_folds}...")
            
            # Create train/test subsets
            train_subset = Subset(self.dataset, train_indices)
            test_subset = Subset(self.dataset, test_indices)
            
            # Create dataloader for test set
            test_loader = DataLoader(test_subset, batch_size=self.config.get('batch_size', 32), num_workers=0)
            
            # Clone the model for this fold
            fold_model = type(self.model)(*self.model.__init__.__defaults__ or ())
            fold_model.load_state_dict(self.model.state_dict())
            fold_model = fold_model.to(self.device)
            fold_model.eval()
            
            # Evaluate on test set
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = fold_model(inputs)
                    
                    # Handle different output shapes
                    if outputs.dim() == 1 or outputs.shape[1] == 1:
                        # For regression or ranking tasks
                        predictions = outputs.view(-1)
                        if targets.dim() > 1:
                            targets = targets.view(-1)
                        errors = (predictions.round() != targets).cpu().numpy()
                    else:
                        # For classification tasks
                        predictions = outputs.argmax(dim=1)
                        if targets.dim() > 1:
                            targets = targets.argmax(dim=1)
                        errors = (predictions != targets).cpu().numpy()
                    
                    # Increment error counts for misclassified samples
                    for i, idx in enumerate(test_indices[:len(errors)]):
                        if errors[i]:
                            error_counts[idx] += 1
        
        # Calculate error rates and find consistent errors
        error_rates = error_counts / n_folds
        
        # Sort samples by error rate in descending order
        sorted_indices = np.argsort(error_rates)[::-1]
        error_indices = []
        error_rates_values = []
        
        for idx in sorted_indices:
            if error_rates[idx] > 0:
                error_indices.append(int(idx))
                error_rates_values.append(float(error_rates[idx]))
        
        # Store results
        result = {
            'error_indices': error_indices,
            'error_rates': error_rates_values,
            'n_folds': n_folds
        }
        self.results['consistent_errors'] = result
        print(f"Found {len(error_indices)} samples with consistent errors.")
        
        return result
    
    def detect_feature_space_outliers(self, method='isolation_forest', **kwargs):
        """識別特徵空間中的離群點。
        
        Args:
            method: 離群點檢測方法，支持'isolation_forest', 'lof', 'dbscan'等
            **kwargs: 傳遞給離群點檢測算法的參數
            
        Returns:
            dict: 包含離群樣本索引及其離群分數
        """
        print(f"Detecting feature space outliers using {method}...")
        
        # Extract features for all samples
        features = self._extract_features()
        
        # Choose outlier detection method
        if method == 'isolation_forest':
            n_estimators = kwargs.get('n_estimators', 100)
            contamination = kwargs.get('contamination', 0.05)
            
            # Train Isolation Forest
            detector = IsolationForest(n_estimators=n_estimators, 
                                      contamination=contamination,
                                      random_state=kwargs.get('random_state', 42))
            
            # Fit and predict
            scores = detector.fit_predict(features)
            outlier_scores = detector.decision_function(features)
            
            # Identify outliers (scores of -1)
            outlier_indices = [i for i, s in enumerate(scores) if s == -1]
            outlier_score_values = [-s for i, s in enumerate(outlier_scores) if i in outlier_indices]
            
        elif method == 'lof':
            n_neighbors = kwargs.get('n_neighbors', 20)
            contamination = kwargs.get('contamination', 0.05)
            
            # Train Local Outlier Factor
            detector = LocalOutlierFactor(n_neighbors=n_neighbors, 
                                         contamination=contamination,
                                         novelty=False)
            
            # Fit and predict
            scores = detector.fit_predict(features)
            
            # Negative outlier factors (higher means more anomalous)
            outlier_scores = -detector.negative_outlier_factor_
            
            # Identify outliers (scores of -1)
            outlier_indices = [i for i, s in enumerate(scores) if s == -1]
            outlier_score_values = [outlier_scores[i] for i in outlier_indices]
            
        elif method == 'dbscan':
            eps = kwargs.get('eps', 0.5)
            min_samples = kwargs.get('min_samples', 5)
            
            # Train DBSCAN
            detector = DBSCAN(eps=eps, min_samples=min_samples)
            
            # Fit and predict
            clusters = detector.fit_predict(features)
            
            # Identify outliers (cluster label of -1)
            outlier_indices = [i for i, c in enumerate(clusters) if c == -1]
            
            # For DBSCAN, we don't have scores, so just use 1 for all outliers
            outlier_score_values = [1.0] * len(outlier_indices)
            
        else:
            raise ValueError(f"Unsupported outlier detection method: {method}")
        
        # Store results
        result = {
            'outlier_indices': outlier_indices,
            'outlier_scores': outlier_score_values,
            'method': method,
            'parameters': kwargs
        }
        self.results['feature_space_outliers'] = result
        print(f"Found {len(outlier_indices)} feature space outliers.")
        
        return result
    
    def _extract_features(self):
        """提取樣本的特徵表示。
        
        Returns:
            numpy.ndarray: 特徵矩陣，形狀為 (n_samples, n_features)
        """
        # Set model to evaluation mode
        self.model.eval()
        self.model.to(self.device)
        
        # Use DataLoader to process in batches
        batch_size = self.config.get('batch_size', 32)
        dataloader = DataLoader(
            self.dataset, 
            batch_size=batch_size,
            shuffle=False, 
            num_workers=0
        )
        
        # Extract features
        features = []
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(self.device)
                
                # For CNN models, we want to extract features before the final layer
                # This is a common pattern, but may need to be adjusted for specific models
                if hasattr(self.model, 'features') and hasattr(self.model, 'classifier'):
                    # This pattern works for many CNN architectures
                    x = self.model.features(inputs)
                    if hasattr(self.model, 'adaptive_pool'):
                        x = self.model.adaptive_pool(x)
                    x = x.view(x.size(0), -1)
                    batch_features = x.cpu().numpy()
                else:
                    # Fallback: use the model outputs as features
                    outputs = self.model(inputs)
                    if outputs.dim() > 1:
                        # Use logits for classification models
                        batch_features = outputs.cpu().numpy()
                    else:
                        # For regression/ranking models, reshape to 2D
                        batch_features = outputs.view(-1, 1).cpu().numpy()
                
                features.append(batch_features)
        
        # Combine batches
        features = np.vstack(features)
        
        return features
    
    def run_comprehensive_detection(self):
        """執行全面的問題數據檢測，結合多種方法。
        
        Returns:
            dict: 綜合檢測結果
        """
        print("Running comprehensive problem data detection...")
        
        # Run gradient anomaly detection if not already done
        if 'gradient_anomalies' not in self.results:
            self.detect_gradient_anomalies()
        
        # Run feature space outlier detection if not already done
        if 'feature_space_outliers' not in self.results:
            self.detect_feature_space_outliers()
        
        # Run consistent error detection if not already done
        if 'consistent_errors' not in self.results:
            self.detect_consistent_errors()
        
        # Combine and return all results
        return self.results
    
    def get_problem_samples_ranking(self):
        """基於多種指標對樣本進行問題程度排名。
        
        Returns:
            list: 按問題程度排序的樣本索引列表
        """
        print("Ranking problem samples...")
        
        # Ensure we have run comprehensive detection
        if not all(k in self.results for k in ['gradient_anomalies', 'feature_space_outliers', 'consistent_errors']):
            self.run_comprehensive_detection()
        
        # Collect all problem samples and their scores
        problem_scores = {}
        
        # Add gradient anomalies with their scores
        if 'gradient_anomalies' in self.results:
            max_grad = max(self.results['gradient_anomalies']['gradient_values']) if self.results['gradient_anomalies']['gradient_values'] else 1.0
            for idx, grad in zip(self.results['gradient_anomalies']['anomaly_indices'], 
                                self.results['gradient_anomalies']['gradient_values']):
                if idx not in problem_scores:
                    problem_scores[idx] = {}
                problem_scores[idx]['gradient_score'] = grad / max_grad
        
        # Add feature space outliers with their scores
        if 'feature_space_outliers' in self.results:
            max_outlier = max(self.results['feature_space_outliers']['outlier_scores']) if self.results['feature_space_outliers']['outlier_scores'] else 1.0
            for idx, score in zip(self.results['feature_space_outliers']['outlier_indices'], 
                                 self.results['feature_space_outliers']['outlier_scores']):
                if idx not in problem_scores:
                    problem_scores[idx] = {}
                problem_scores[idx]['outlier_score'] = score / max_outlier
        
        # Add consistent errors with their scores
        if 'consistent_errors' in self.results:
            for idx, rate in zip(self.results['consistent_errors']['error_indices'], 
                                self.results['consistent_errors']['error_rates']):
                if idx not in problem_scores:
                    problem_scores[idx] = {}
                problem_scores[idx]['error_rate'] = rate
        
        # Calculate combined scores for each sample
        combined_scores = []
        for idx, scores in problem_scores.items():
            # Calculate average score across all metrics (present for this sample)
            avg_score = np.mean(list(scores.values()))
            combined_scores.append((idx, avg_score))
        
        # Sort by combined score in descending order
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return sorted list of indices
        return [idx for idx, _ in combined_scores]
    
    def save_results(self, filepath):
        """將檢測結果保存到文件。
        
        Args:
            filepath: 保存路徑
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert numpy values to native Python types for JSON serialization
        results_to_save = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                results_to_save[key] = {}
                for k, v in value.items():
                    if isinstance(v, (np.ndarray, list)):
                        if isinstance(v, np.ndarray):
                            v = v.tolist()
                        results_to_save[key][k] = [float(x) if isinstance(x, (np.float32, np.float64)) else 
                                                 int(x) if isinstance(x, (np.int32, np.int64)) else x 
                                                 for x in v]
                    elif isinstance(v, (np.float32, np.float64)):
                        results_to_save[key][k] = float(v)
                    elif isinstance(v, (np.int32, np.int64)):
                        results_to_save[key][k] = int(v)
                    else:
                        results_to_save[key][k] = v
            else:
                results_to_save[key] = value
        
        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def load_results(self, filepath):
        """從文件加載檢測結果。
        
        Args:
            filepath: 文件路徑
            
        Returns:
            dict: 加載的檢測結果
        """
        # Load from JSON
        with open(filepath, 'r') as f:
            self.results = json.load(f)
        
        print(f"Results loaded from {filepath}")
        return self.results 