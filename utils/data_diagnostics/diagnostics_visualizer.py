"""
Diagnostics visualizer module.

This module provides visualization tools for analyzing problem data,
including feature space visualization, gradient distribution, error patterns,
and quality metrics visualization.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json
import tempfile
from datetime import datetime
from torch.utils.data import DataLoader

try:
    # Try to import UMAP for better visualization
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    # Try to import plotly for interactive visualization
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

class DiagnosticsVisualizer:
    """問題數據可視化工具，提供多種方法來可視化問題樣本的特性和分布。"""
    
    def __init__(self, detector, dataset, config=None):
        """初始化可視化器。
        
        Args:
            detector: ProblemSampleDetector 實例，包含已檢測的問題樣本
            dataset: 數據集
            config: 可視化配置
        """
        self.detector = detector
        self.dataset = dataset
        self.config = config or {}
        self.output_dir = self.config.get('output_dir', './visualization_results')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def visualize_feature_space(self, method='tsne', dims=2, sample_indices=None, **kwargs):
        """在降維後的特徵空間中可視化數據樣本。
        
        Args:
            method: 降維方法，支持'tsne', 'pca', 'umap'等
            dims: 降維後的維度（通常為2或3）
            sample_indices: 要可視化的樣本索引，None表示全部
            **kwargs: 傳遞給降維算法的參數
            
        Returns:
            matplotlib.figure.Figure: 生成的圖形
        """
        print(f"Visualizing feature space using {method} with {dims} dimensions...")
        
        # Extract features from samples
        features, labels = self._extract_features_and_labels(sample_indices)
        
        # Apply dimensionality reduction
        if method.lower() == 'tsne':
            # Default parameters for t-SNE
            perplexity = kwargs.get('perplexity', 30)
            learning_rate = kwargs.get('learning_rate', 200)
            n_iter = kwargs.get('n_iter', 1000)
            
            # Apply t-SNE
            tsne = TSNE(n_components=dims, perplexity=perplexity, 
                        learning_rate=learning_rate, n_iter=n_iter, random_state=42)
            reduced_features = tsne.fit_transform(features)
            
        elif method.lower() == 'pca':
            # Apply PCA
            pca = PCA(n_components=dims, random_state=42)
            reduced_features = pca.fit_transform(features)
            
        elif method.lower() == 'umap' and UMAP_AVAILABLE:
            # Default parameters for UMAP
            n_neighbors = kwargs.get('n_neighbors', 15)
            min_dist = kwargs.get('min_dist', 0.1)
            
            # Apply UMAP
            umap = UMAP(n_components=dims, n_neighbors=n_neighbors, 
                       min_dist=min_dist, random_state=42)
            reduced_features = umap.fit_transform(features)
            
        else:
            if method.lower() == 'umap' and not UMAP_AVAILABLE:
                print("UMAP not available. Falling back to t-SNE.")
                method = 'tsne'
                tsne = TSNE(n_components=dims, random_state=42)
                reduced_features = tsne.fit_transform(features)
            else:
                raise ValueError(f"Unsupported dimensionality reduction method: {method}")
        
        # Get problem sample indices from detector
        problem_indices = self._get_problem_indices()
        
        # Create a mask for problem samples
        if sample_indices is not None:
            is_problem = np.zeros(len(sample_indices), dtype=bool)
            for i, idx in enumerate(sample_indices):
                is_problem[i] = idx in problem_indices
        else:
            is_problem = np.zeros(len(self.dataset), dtype=bool)
            is_problem[problem_indices] = True
            # Only keep the samples we're actually visualizing
            is_problem = is_problem[:len(features)]
        
        # Create figure
        fig = plt.figure(figsize=(10, 8))
        
        if dims == 2:
            # 2D plot
            ax = fig.add_subplot(111)
            
            # Plot normal samples
            normal_mask = ~is_problem
            if np.any(normal_mask):
                scatter_normal = ax.scatter(
                    reduced_features[normal_mask, 0], 
                    reduced_features[normal_mask, 1],
                    c=labels[normal_mask] if labels is not None else 'blue',
                    alpha=0.6, marker='o', label='Normal Samples'
                )
            
            # Plot problem samples
            if np.any(is_problem):
                scatter_problem = ax.scatter(
                    reduced_features[is_problem, 0], 
                    reduced_features[is_problem, 1],
                    c=labels[is_problem] if labels is not None else 'red',
                    alpha=0.8, marker='X', s=100, label='Problem Samples',
                    edgecolors='black'
                )
            
            # Add colorbar if we have labels
            if labels is not None:
                unique_labels = np.unique(labels)
                if len(unique_labels) <= 10:  # Only add legend for a reasonable number of classes
                    if np.any(normal_mask):
                        plt.colorbar(scatter_normal, label='Class')
            
            ax.set_xlabel(f'{method.upper()} Dimension 1')
            ax.set_ylabel(f'{method.upper()} Dimension 2')
            
        elif dims == 3:
            # 3D plot
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot normal samples
            normal_mask = ~is_problem
            if np.any(normal_mask):
                scatter_normal = ax.scatter(
                    reduced_features[normal_mask, 0], 
                    reduced_features[normal_mask, 1],
                    reduced_features[normal_mask, 2],
                    c=labels[normal_mask] if labels is not None else 'blue',
                    alpha=0.6, marker='o', label='Normal Samples'
                )
            
            # Plot problem samples
            if np.any(is_problem):
                scatter_problem = ax.scatter(
                    reduced_features[is_problem, 0], 
                    reduced_features[is_problem, 1],
                    reduced_features[is_problem, 2],
                    c=labels[is_problem] if labels is not None else 'red',
                    alpha=0.8, marker='X', s=100, label='Problem Samples',
                    edgecolors='black'
                )
            
            # Add colorbar if we have labels
            if labels is not None:
                unique_labels = np.unique(labels)
                if len(unique_labels) <= 10:  # Only add legend for a reasonable number of classes
                    if np.any(normal_mask):
                        plt.colorbar(scatter_normal, label='Class')
            
            ax.set_xlabel(f'{method.upper()} Dimension 1')
            ax.set_ylabel(f'{method.upper()} Dimension 2')
            ax.set_zlabel(f'{method.upper()} Dimension 3')
        
        plt.title(f'Feature Space Visualization ({method.upper()}) - Problem Samples Highlighted')
        plt.legend()
        plt.tight_layout()
        
        return fig
    
    def visualize_gradient_distribution(self, gradient_data=None, threshold=0.95, bins=50):
        """可視化樣本梯度分布，標記異常值。
        
        Args:
            gradient_data: 梯度數據，None則使用detector中的結果
            threshold: 異常判定閾值
            bins: 直方圖的分箱數
            
        Returns:
            matplotlib.figure.Figure: 生成的圖形
        """
        print(f"Visualizing gradient distribution with threshold {threshold}...")
        
        # Get gradient data
        if gradient_data is None:
            if 'gradient_anomalies' not in self.detector.results:
                print("No gradient data found in detector. Running gradient anomaly detection...")
                self.detector.detect_gradient_anomalies(threshold=threshold)
            
            # Get sample gradients
            all_gradients = []
            indices = []
            dataloader = DataLoader(
                self.dataset, batch_size=32, shuffle=False, num_workers=0
            )
            
            # Set model to evaluation mode
            self.detector.model.eval()
            
            # Compute gradients for all samples
            criterion = torch.nn.CrossEntropyLoss(reduction='none')
            device = self.detector.device
            
            batch_idx = 0
            for inputs, targets in dataloader:
                batch_indices = list(range(batch_idx * 32, min((batch_idx + 1) * 32, len(self.dataset))))
                indices.extend(batch_indices)
                
                # Move data to device
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Enable gradient tracking
                inputs.requires_grad = True
                
                # Forward pass
                outputs = self.detector.model(inputs)
                
                # Compute loss
                if outputs.dim() == 1 or outputs.shape[1] == 1:
                    outputs = outputs.view(-1, 1)
                    if targets.dim() == 1:
                        targets = targets.float().view(-1, 1)
                    loss = torch.nn.functional.mse_loss(outputs, targets.float(), reduction='none')
                else:
                    if targets.dim() == 2:
                        targets = targets.argmax(dim=1)
                    loss = criterion(outputs, targets)
                
                # Get gradients for each sample
                for i in range(len(inputs)):
                    if inputs.grad is not None:
                        inputs.grad.zero_()
                    
                    if loss.dim() > 0:
                        sample_loss = loss[i].mean()
                    else:
                        sample_loss = loss
                    
                    sample_loss.backward(retain_graph=True)
                    
                    if inputs.grad is not None:
                        grad_magnitude = inputs.grad[i].abs().mean().item()
                        all_gradients.append(grad_magnitude)
                
                # Free memory
                inputs.requires_grad = False
                inputs.grad = None
                
                batch_idx += 1
            
            # Create gradient data
            gradient_data = {'gradients': all_gradients, 'indices': indices}
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram
        counts, bin_edges, patches = ax.hist(gradient_data['gradients'], bins=bins, alpha=0.7)
        
        # Calculate threshold value
        threshold_value = np.percentile(gradient_data['gradients'], threshold * 100)
        
        # Mark anomalies in red
        for i, patch in enumerate(patches):
            if bin_edges[i] >= threshold_value:
                patch.set_facecolor('red')
        
        # Add threshold line
        ax.axvline(x=threshold_value, color='r', linestyle='--', 
                  label=f'Threshold ({threshold:.2f} percentile)')
        
        ax.set_xlabel('Gradient Magnitude')
        ax.set_ylabel('Count')
        ax.set_title('Gradient Distribution with Anomalies Highlighted')
        ax.legend()
        plt.tight_layout()
        
        return fig
    
    def _extract_features_and_labels(self, sample_indices=None):
        """Extract features and labels from dataset samples.
        
        Args:
            sample_indices: Specific sample indices to use, None for all
            
        Returns:
            tuple: (features, labels)
        """
        # Convert indices to list if needed
        if sample_indices is None:
            sample_indices = list(range(len(self.dataset)))
        
        # Create subset dataset if needed
        if len(sample_indices) < len(self.dataset):
            subset = torch.utils.data.Subset(self.dataset, sample_indices)
            dataloader = DataLoader(subset, batch_size=32, shuffle=False, num_workers=0)
        else:
            dataloader = DataLoader(self.dataset, batch_size=32, shuffle=False, num_workers=0)
        
        # Extract features and labels
        features = []
        labels = []
        device = self.detector.device
        
        # Set model to evaluation mode
        self.detector.model.eval()
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                # Move to device
                inputs = inputs.to(device)
                
                # Extract features
                if hasattr(self.detector.model, 'features') and hasattr(self.detector.model, 'classifier'):
                    # For CNN models
                    x = self.detector.model.features(inputs)
                    if hasattr(self.detector.model, 'adaptive_pool'):
                        x = self.detector.model.adaptive_pool(x)
                    batch_features = x.view(x.size(0), -1).cpu().numpy()
                else:
                    # Fallback: use model outputs
                    outputs = self.detector.model(inputs)
                    if outputs.dim() > 1:
                        batch_features = outputs.cpu().numpy()
                    else:
                        batch_features = outputs.view(-1, 1).cpu().numpy()
                
                # Add to lists
                features.append(batch_features)
                
                # Get labels
                if targets.dim() > 1:
                    # Get class index from one-hot
                    batch_labels = targets.argmax(dim=1).cpu().numpy()
                else:
                    batch_labels = targets.cpu().numpy()
                
                labels.append(batch_labels)
        
        # Combine batches
        features = np.vstack(features) if features else np.array([])
        labels = np.concatenate(labels) if labels else np.array([])
        
        return features, labels
    
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
    
    def visualize_error_patterns(self, error_data=None, top_n=10):
        """可視化樣本錯誤模式和類型。
        
        Args:
            error_data: 錯誤數據，None則使用detector中的結果
            top_n: 顯示前N個最常見錯誤類型
            
        Returns:
            matplotlib.figure.Figure: 生成的圖形
        """
        print(f"Visualizing error patterns, showing top {top_n} patterns...")
        
        # Get error data
        if error_data is None:
            if 'consistent_errors' not in self.detector.results:
                print("No error data found in detector. Running consistent error detection...")
                self.detector.detect_consistent_errors()
            
            # Get error indices and rates
            error_indices = self.detector.results['consistent_errors']['error_indices']
            error_rates = self.detector.results['consistent_errors']['error_rates']
        else:
            error_indices = error_data.get('error_indices', [])
            error_rates = error_data.get('error_rates', [])
        
        # Get model predictions for error samples
        error_predictions = {}
        
        # Set model to evaluation mode
        self.detector.model.eval()
        device = self.detector.device
        
        # Process in batches for efficiency
        batch_size = 32
        for i in range(0, len(error_indices), batch_size):
            batch_indices = error_indices[i:i+batch_size]
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
            
            # Get predictions
            with torch.no_grad():
                outputs = self.detector.model(inputs)
                
                # Handle different output shapes
                if outputs.dim() == 1 or outputs.shape[1] == 1:
                    # For regression or ranking tasks
                    predictions = outputs.view(-1).round().cpu().numpy()
                    if targets.dim() > 1:
                        targets = targets.view(-1)
                    true_labels = targets.cpu().numpy()
                else:
                    # For classification tasks
                    predictions = outputs.argmax(dim=1).cpu().numpy()
                    if targets.dim() > 1:
                        targets = targets.argmax(dim=1)
                    true_labels = targets.cpu().numpy()
            
            # Record error patterns
            for j, (pred, true, idx) in enumerate(zip(predictions, true_labels, batch_indices)):
                error_type = f"{true} → {pred}"
                if error_type not in error_predictions:
                    error_predictions[error_type] = {'count': 0, 'indices': []}
                
                error_predictions[error_type]['count'] += 1
                error_predictions[error_type]['indices'].append(idx)
        
        # Sort error patterns by frequency
        sorted_errors = sorted(error_predictions.items(), key=lambda x: x[1]['count'], reverse=True)
        
        # Create figure for error patterns
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot error rates distribution
        ax1.hist(error_rates, bins=10, alpha=0.7, color='coral')
        ax1.set_xlabel('Error Rate')
        ax1.set_ylabel('Count')
        ax1.set_title('Distribution of Error Rates')
        
        # Plot top error patterns
        top_errors = sorted_errors[:min(top_n, len(sorted_errors))]
        error_types = [e[0] for e in top_errors]
        error_counts = [e[1]['count'] for e in top_errors]
        
        # Horizontal bar chart for error patterns
        y_pos = np.arange(len(error_types))
        ax2.barh(y_pos, error_counts, align='center', alpha=0.7, color='lightblue')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(error_types)
        ax2.invert_yaxis()  # Labels read top-to-bottom
        ax2.set_xlabel('Count')
        ax2.set_title(f'Top {len(top_errors)} Error Patterns')
        
        plt.tight_layout()
        
        return fig
    
    def visualize_quality_metrics(self, metrics_data=None, sample_indices=None):
        """可視化樣本質量指標的分布。
        
        Args:
            metrics_data: 質量指標數據，None則使用detector中的結果
            sample_indices: 要可視化的樣本索引，None表示全部
            
        Returns:
            matplotlib.figure.Figure: 生成的圖形
        """
        print("Visualizing quality metrics...")
        
        # Get quality metrics data
        if metrics_data is None:
            # We need to compute quality metrics for samples
            # This is a simplified version that uses existing detector results
            metrics_data = {}
            
            # Get problem samples
            problem_indices = self._get_problem_indices()
            
            # Limit to sample_indices if provided
            if sample_indices is not None:
                indices_to_process = set(sample_indices).intersection(problem_indices)
            else:
                indices_to_process = problem_indices
            
            # Add gradient anomaly scores
            if 'gradient_anomalies' in self.detector.results:
                for idx, grad in zip(self.detector.results['gradient_anomalies']['anomaly_indices'],
                                   self.detector.results['gradient_anomalies']['gradient_values']):
                    if idx in indices_to_process:
                        if idx not in metrics_data:
                            metrics_data[idx] = {}
                        metrics_data[idx]['gradient_score'] = grad
            
            # Add outlier scores
            if 'feature_space_outliers' in self.detector.results:
                for idx, score in zip(self.detector.results['feature_space_outliers']['outlier_indices'],
                                    self.detector.results['feature_space_outliers']['outlier_scores']):
                    if idx in indices_to_process:
                        if idx not in metrics_data:
                            metrics_data[idx] = {}
                        metrics_data[idx]['outlier_score'] = score
            
            # Add error rates
            if 'consistent_errors' in self.detector.results:
                for idx, rate in zip(self.detector.results['consistent_errors']['error_indices'],
                                   self.detector.results['consistent_errors']['error_rates']):
                    if idx in indices_to_process:
                        if idx not in metrics_data:
                            metrics_data[idx] = {}
                        metrics_data[idx]['error_rate'] = rate
        
        # Check if we have data to visualize
        if not metrics_data:
            print("No quality metrics data available.")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "No quality metrics data available", 
                   ha='center', va='center', fontsize=12)
            return fig
        
        # Extract metrics and normalize
        metrics_by_type = {}
        for idx, metrics in metrics_data.items():
            for metric_name, value in metrics.items():
                if metric_name not in metrics_by_type:
                    metrics_by_type[metric_name] = []
                metrics_by_type[metric_name].append(value)
        
        # Normalize metrics to [0, 1] range
        normalized_metrics = {}
        for metric_name, values in metrics_by_type.items():
            min_val = min(values)
            max_val = max(values)
            if max_val > min_val:
                normalized_metrics[metric_name] = [(v - min_val) / (max_val - min_val) for v in values]
            else:
                normalized_metrics[metric_name] = [0.5 for _ in values]  # Default value if all metrics are the same
        
        # Create figure
        n_metrics = len(metrics_by_type)
        if n_metrics == 0:
            print("No metrics available to visualize.")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "No metrics available to visualize", 
                   ha='center', va='center', fontsize=12)
            return fig
        
        # Create subplots based on number of metrics
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4 * n_metrics))
        if n_metrics == 1:
            axes = [axes]  # Make it iterable
        
        # Plot distributions for each metric
        for i, (metric_name, values) in enumerate(metrics_by_type.items()):
            ax = axes[i]
            ax.hist(values, bins=20, alpha=0.7, color=f'C{i}')
            ax.set_xlabel(metric_name)
            ax.set_ylabel('Count')
            ax.set_title(f'Distribution of {metric_name}')
            
            # Add mean/median lines
            mean_val = np.mean(values)
            median_val = np.median(values)
            ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
            ax.axvline(median_val, color='green', linestyle=':', label=f'Median: {median_val:.3f}')
            ax.legend()
        
        plt.tight_layout()
        
        return fig
    
    def generate_interactive_visualization(self, output_type='html', **kwargs):
        """生成交互式可視化（使用plotly或bokeh等）。
        
        Args:
            output_type: 輸出類型，'html'或'notebook'
            **kwargs: 可視化參數
            
        Returns:
            visualization: 交互式可視化物件或文件路徑
        """
        print(f"Generating interactive visualization with output type: {output_type}...")
        
        if not PLOTLY_AVAILABLE:
            print("Plotly is not available. Interactive visualization requires plotly.")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "Plotly is required for interactive visualization", 
                   ha='center', va='center', fontsize=12)
            
            # Save static image instead
            filepath = os.path.join(self.output_dir, "interactive_viz_fallback.png")
            plt.savefig(filepath)
            plt.close(fig)
            
            return filepath
        
        # Extract features and labels
        features, labels = self._extract_features_and_labels()
        
        # Apply dimensionality reduction (default to PCA)
        method = kwargs.get('method', 'pca')
        dims = kwargs.get('dims', 2)
        
        if method.lower() == 'tsne':
            tsne = TSNE(n_components=dims, random_state=42)
            reduced_features = tsne.fit_transform(features)
        elif method.lower() == 'umap' and UMAP_AVAILABLE:
            umap = UMAP(n_components=dims, random_state=42)
            reduced_features = umap.fit_transform(features)
        else:
            # Default to PCA
            pca = PCA(n_components=dims, random_state=42)
            reduced_features = pca.fit_transform(features)
        
        # Get problem sample indices
        problem_indices = self._get_problem_indices()
        
        # Create mask for problem samples
        is_problem = np.zeros(len(self.dataset), dtype=bool)
        is_problem[problem_indices] = True
        is_problem = is_problem[:len(features)]  # Ensure same length as features
        
        # Create plotly figure
        if dims == 2:
            # Create dataframe for plotly express
            df = {
                'x': reduced_features[:, 0],
                'y': reduced_features[:, 1],
                'label': labels,
                'is_problem': is_problem
            }
            
            # Use plotly express for scatter plot
            fig = px.scatter(
                df, x='x', y='y', color='label', 
                symbol='is_problem',
                symbol_map={False: 'circle', True: 'x'},
                color_continuous_scale=px.colors.qualitative.Plotly,
                title=f'Interactive Feature Space Visualization ({method.upper()})',
                labels={'x': f'{method.upper()} Dimension 1', 'y': f'{method.upper()} Dimension 2', 
                       'label': 'Class', 'is_problem': 'Problem Sample'}
            )
            
            # Customize hover info
            hovertemplate = (
                "Index: %{customdata}<br>" +
                "Class: %{color}<br>" +
                "Problem Sample: %{symbol}"
            )
            
            fig.update_traces(
                customdata=np.arange(len(features)),
                hovertemplate=hovertemplate,
                marker=dict(size=10, line=dict(width=1, color='black')),
            )
            
        elif dims == 3:
            # Create 3D scatter plot
            fig = go.Figure(data=[
                go.Scatter3d(
                    x=reduced_features[~is_problem, 0],
                    y=reduced_features[~is_problem, 1],
                    z=reduced_features[~is_problem, 2],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=labels[~is_problem],
                        colorscale='Viridis',
                        opacity=0.8
                    ),
                    customdata=np.arange(len(features))[~is_problem],
                    hovertemplate=(
                        "Index: %{customdata}<br>" +
                        "Class: %{marker.color}<br>" +
                        "Normal Sample"
                    ),
                    name='Normal Samples'
                ),
                go.Scatter3d(
                    x=reduced_features[is_problem, 0],
                    y=reduced_features[is_problem, 1],
                    z=reduced_features[is_problem, 2],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=labels[is_problem],
                        colorscale='Viridis',
                        symbol='cross',
                        opacity=1,
                        line=dict(width=1, color='black')
                    ),
                    customdata=np.arange(len(features))[is_problem],
                    hovertemplate=(
                        "Index: %{customdata}<br>" +
                        "Class: %{marker.color}<br>" +
                        "Problem Sample"
                    ),
                    name='Problem Samples'
                )
            ])
            
            fig.update_layout(
                title=f'Interactive 3D Feature Space Visualization ({method.upper()})',
                scene=dict(
                    xaxis_title=f'{method.upper()} Dimension 1',
                    yaxis_title=f'{method.upper()} Dimension 2',
                    zaxis_title=f'{method.upper()} Dimension 3',
                ),
            )
        
        # Handle output
        if output_type.lower() == 'html':
            # Save as HTML file
            filename = f"interactive_visualization_{method}_{dims}d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = os.path.join(self.output_dir, filename)
            fig.write_html(filepath)
            return filepath
        else:
            # Return plotly figure for notebook display
            return fig
    
    def generate_comprehensive_report(self, filename='diagnostic_report.html'):
        """生成綜合診斷報告，包含多個可視化結果。
        
        Args:
            filename: 報告文件名
            
        Returns:
            str: 報告文件路徑
        """
        print(f"Generating comprehensive diagnostic report: {filename}")
        
        if not PLOTLY_AVAILABLE:
            print("Warning: Plotly is not available. Report will use static images.")
        
        # Generate visualizations
        feature_viz = self.visualize_feature_space(method='pca')
        gradient_viz = self.visualize_gradient_distribution()
        error_viz = self.visualize_error_patterns()
        quality_viz = self.visualize_quality_metrics()
        
        # Save visualizations as image files
        feature_path = self.save_visualization(feature_viz, "feature_space.png")
        gradient_path = self.save_visualization(gradient_viz, "gradient_distribution.png")
        error_path = self.save_visualization(error_viz, "error_patterns.png")
        quality_path = self.save_visualization(quality_viz, "quality_metrics.png")
        
        # Generate interactive visualization if plotly is available
        if PLOTLY_AVAILABLE:
            interactive_path = self.generate_interactive_visualization(output_type='html')
        else:
            interactive_path = None
        
        # Create HTML report
        report_path = os.path.join(self.output_dir, filename)
        
        with open(report_path, 'w') as f:
            f.write('<html>\n')
            f.write('<head>\n')
            f.write('  <title>Data Diagnostics Report</title>\n')
            f.write('  <style>\n')
            f.write('    body { font-family: Arial, sans-serif; margin: 20px; }\n')
            f.write('    h1 { color: #2c3e50; }\n')
            f.write('    h2 { color: #3498db; margin-top: 30px; }\n')
            f.write('    .section { margin-bottom: 40px; }\n')
            f.write('    .viz-container { max-width: 1000px; margin: 20px 0; }\n')
            f.write('    table { border-collapse: collapse; width: 100%; }\n')
            f.write('    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n')
            f.write('    th { background-color: #f2f2f2; }\n')
            f.write('    tr:nth-child(even) { background-color: #f9f9f9; }\n')
            f.write('  </style>\n')
            f.write('</head>\n')
            f.write('<body>\n')
            
            # Header
            f.write('  <h1>Data Diagnostics Report</h1>\n')
            f.write(f'  <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>\n')
            
            # Summary
            f.write('  <div class="section">\n')
            f.write('    <h2>Summary</h2>\n')
            
            # Calculate summary statistics
            problem_indices = self._get_problem_indices()
            n_problems = len(problem_indices)
            n_samples = len(self.dataset)
            problem_percentage = (n_problems / n_samples) * 100 if n_samples > 0 else 0
            
            # Count by problem type
            n_gradient = len(self.detector.results.get('gradient_anomalies', {}).get('anomaly_indices', []))
            n_outliers = len(self.detector.results.get('feature_space_outliers', {}).get('outlier_indices', []))
            n_errors = len(self.detector.results.get('consistent_errors', {}).get('error_indices', []))
            
            f.write('    <table>\n')
            f.write('      <tr><th>Metric</th><th>Value</th></tr>\n')
            f.write(f'      <tr><td>Total Samples</td><td>{n_samples}</td></tr>\n')
            f.write(f'      <tr><td>Problem Samples</td><td>{n_problems} ({problem_percentage:.2f}%)</td></tr>\n')
            f.write(f'      <tr><td>Gradient Anomalies</td><td>{n_gradient}</td></tr>\n')
            f.write(f'      <tr><td>Feature Space Outliers</td><td>{n_outliers}</td></tr>\n')
            f.write(f'      <tr><td>Consistent Errors</td><td>{n_errors}</td></tr>\n')
            f.write('    </table>\n')
            f.write('  </div>\n')
            
            # Feature Space Visualization
            f.write('  <div class="section">\n')
            f.write('    <h2>Feature Space Visualization</h2>\n')
            f.write('    <p>Visualization of samples in the feature space, with problem samples highlighted.</p>\n')
            f.write('    <div class="viz-container">\n')
            f.write(f'      <img src="{os.path.basename(feature_path)}" style="max-width:100%;" />\n')
            f.write('    </div>\n')
            f.write('  </div>\n')
            
            # Gradient Distribution
            f.write('  <div class="section">\n')
            f.write('    <h2>Gradient Distribution</h2>\n')
            f.write('    <p>Distribution of gradient magnitudes, with anomalies highlighted in red.</p>\n')
            f.write('    <div class="viz-container">\n')
            f.write(f'      <img src="{os.path.basename(gradient_path)}" style="max-width:100%;" />\n')
            f.write('    </div>\n')
            f.write('  </div>\n')
            
            # Error Patterns
            f.write('  <div class="section">\n')
            f.write('    <h2>Error Patterns</h2>\n')
            f.write('    <p>Analysis of common error patterns in the dataset.</p>\n')
            f.write('    <div class="viz-container">\n')
            f.write(f'      <img src="{os.path.basename(error_path)}" style="max-width:100%;" />\n')
            f.write('    </div>\n')
            f.write('  </div>\n')
            
            # Quality Metrics
            f.write('  <div class="section">\n')
            f.write('    <h2>Quality Metrics</h2>\n')
            f.write('    <p>Distribution of various quality metrics for the dataset.</p>\n')
            f.write('    <div class="viz-container">\n')
            f.write(f'      <img src="{os.path.basename(quality_path)}" style="max-width:100%;" />\n')
            f.write('    </div>\n')
            f.write('  </div>\n')
            
            # Interactive Visualization
            if interactive_path:
                f.write('  <div class="section">\n')
                f.write('    <h2>Interactive Visualization</h2>\n')
                f.write('    <p>Interactive visualization of the feature space.</p>\n')
                f.write('    <p><a href="' + os.path.basename(interactive_path) + '" target="_blank">Open Interactive Visualization</a></p>\n')
                f.write('  </div>\n')
            
            # Problem Samples List
            f.write('  <div class="section">\n')
            f.write('    <h2>Top Problem Samples</h2>\n')
            f.write('    <p>List of the most problematic samples in the dataset.</p>\n')
            
            # Get ranked problem samples
            ranked_samples = self.detector.get_problem_samples_ranking()
            top_samples = ranked_samples[:min(20, len(ranked_samples))]
            
            if top_samples:
                f.write('    <table>\n')
                f.write('      <tr><th>Rank</th><th>Sample Index</th><th>Problem Types</th></tr>\n')
                
                for rank, idx in enumerate(top_samples):
                    problem_types = []
                    
                    if 'gradient_anomalies' in self.detector.results and idx in self.detector.results['gradient_anomalies']['anomaly_indices']:
                        problem_types.append('Gradient Anomaly')
                    
                    if 'feature_space_outliers' in self.detector.results and idx in self.detector.results['feature_space_outliers']['outlier_indices']:
                        problem_types.append('Feature Space Outlier')
                    
                    if 'consistent_errors' in self.detector.results and idx in self.detector.results['consistent_errors']['error_indices']:
                        problem_types.append('Consistent Error')
                    
                    f.write(f'      <tr><td>{rank+1}</td><td>{idx}</td><td>{", ".join(problem_types)}</td></tr>\n')
                
                f.write('    </table>\n')
            else:
                f.write('    <p>No problem samples found.</p>\n')
            
            f.write('  </div>\n')
            
            f.write('</body>\n')
            f.write('</html>\n')
        
        print(f"Report generated at: {report_path}")
        return report_path
    
    def save_visualization(self, fig, filename):
        """保存可視化結果到文件。
        
        Args:
            fig: 圖形物件
            filename: 文件名
            
        Returns:
            str: 保存的文件路徑
        """
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return filepath 