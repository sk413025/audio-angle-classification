"""
Data quality metrics module.

This module provides functions to calculate various quality metrics for individual samples,
helping to identify problematic data that might affect model performance.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
import copy

def calculate_sample_difficulty(model, sample, target, topk=(1,)):
    """計算樣本的難度指標。
    
    Args:
        model: 已訓練的模型
        sample: 單個樣本數據
        target: 樣本標籤
        topk: 計算top-k準確率的k值
        
    Returns:
        float: 樣本難度分數，越高表示越難
    """
    # Ensure sample is a tensor with batch dimension
    if not isinstance(sample, torch.Tensor):
        sample = torch.tensor(sample, dtype=torch.float32)
    
    if sample.dim() == 1:
        sample = sample.unsqueeze(0)
    
    # Ensure target is a tensor
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target)
    
    if target.dim() == 0:
        target = target.unsqueeze(0)
    
    # Get device
    device = next(model.parameters()).device
    
    # Move data to device
    sample = sample.to(device)
    target = target.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        output = model(sample)
    
    # Calculate difficulty based on model output
    if output.dim() == 1 or output.shape[1] == 1:
        # For regression or ranking tasks
        output = output.view(-1)
        if target.dim() > 1:
            target = target.view(-1)
        
        # Use mean squared error as difficulty measure
        difficulty = torch.nn.functional.mse_loss(output, target.float()).item()
        
        # Normalize to [0, 1] range (assuming reasonable MSE range)
        difficulty = min(difficulty / 10.0, 1.0)
    else:
        # For classification tasks
        # Calculate prediction probabilities
        probabilities = torch.nn.functional.softmax(output, dim=1)
        
        # Get target class probability
        if target.dim() > 1:
            # If target is one-hot encoded
            target_probs = torch.sum(probabilities * target, dim=1)
        else:
            target_probs = probabilities[torch.arange(probabilities.size(0)), target]
        
        # Higher probability means easier sample, so we invert
        difficulty = 1.0 - target_probs.item()
        
        # Calculate top-k accuracy for additional difficulty measure
        _, pred = output.topk(max(topk), 1, True, True)
        correct = pred.eq(target.view(-1, 1).expand_as(pred))
        
        # If not in top-k, increase difficulty
        for k in topk:
            if not correct[:, :k].any().item():
                # Sample is even more difficult if it's not in top-k
                difficulty = min(difficulty + 0.1 * (1.0 / k), 1.0)
    
    return difficulty

def calculate_sample_influence(model, sample, target, validation_set, criterion):
    """計算樣本對模型性能的影響力。
    
    Args:
        model: 已訓練的模型
        sample: 單個樣本數據
        target: 樣本標籤
        validation_set: 驗證集，用於評估影響
        criterion: 損失函數
        
    Returns:
        float: 樣本影響力分數
    """
    # Ensure sample is a tensor with batch dimension
    if not isinstance(sample, torch.Tensor):
        sample = torch.tensor(sample, dtype=torch.float32)
    
    if sample.dim() == 1:
        sample = sample.unsqueeze(0)
    
    # Ensure target is a tensor
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target)
    
    if target.dim() == 0:
        target = target.unsqueeze(0)
    
    # Get device
    device = next(model.parameters()).device
    
    # Move data to device
    sample = sample.to(device)
    target = target.to(device)
    
    # Clone the model to avoid affecting the original
    model_clone = copy.deepcopy(model)
    model_clone.train()
    
    # Compute validation loss before
    val_loss_before = _compute_validation_loss(model_clone, validation_set, criterion, device)
    
    # Perform a single update step with the sample
    optimizer = torch.optim.SGD(model_clone.parameters(), lr=0.01)
    optimizer.zero_grad()
    
    # Forward pass
    output = model_clone(sample)
    
    # Calculate loss
    if output.dim() == 1 or output.shape[1] == 1:
        # For regression or ranking tasks
        output = output.view(-1)
        if target.dim() > 1:
            target = target.view(-1)
        loss = criterion(output, target.float())
    else:
        # For classification tasks
        if target.dim() > 1:
            # If target is one-hot encoded
            target = target.argmax(dim=1)
        loss = criterion(output, target)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Compute validation loss after
    val_loss_after = _compute_validation_loss(model_clone, validation_set, criterion, device)
    
    # Influence is the change in validation loss
    influence = val_loss_after - val_loss_before
    
    # Free memory
    del model_clone
    
    return influence

def _compute_validation_loss(model, validation_set, criterion, device):
    """Helper function to compute validation loss."""
    model.eval()
    total_loss = 0.0
    count = 0
    
    with torch.no_grad():
        for val_sample, val_target in validation_set:
            # Ensure data is tensors
            if not isinstance(val_sample, torch.Tensor):
                val_sample = torch.tensor(val_sample, dtype=torch.float32)
            
            if not isinstance(val_target, torch.Tensor):
                val_target = torch.tensor(val_target)
            
            # Add batch dimension if necessary
            if val_sample.dim() == 1:
                val_sample = val_sample.unsqueeze(0)
            
            if val_target.dim() == 0:
                val_target = val_target.unsqueeze(0)
            
            # Move to device
            val_sample = val_sample.to(device)
            val_target = val_target.to(device)
            
            # Forward pass
            val_output = model(val_sample)
            
            # Calculate loss
            if val_output.dim() == 1 or val_output.shape[1] == 1:
                # For regression or ranking tasks
                val_output = val_output.view(-1)
                if val_target.dim() > 1:
                    val_target = val_target.view(-1)
                val_loss = criterion(val_output, val_target.float())
            else:
                # For classification tasks
                if val_target.dim() > 1:
                    # If target is one-hot encoded
                    val_target = val_target.argmax(dim=1)
                val_loss = criterion(val_output, val_target)
            
            total_loss += val_loss.item()
            count += 1
    
    return total_loss / count if count > 0 else 0.0

def calculate_feature_space_density(model, sample, dataset, method='knn', k=5):
    """計算樣本在特徵空間的密度或離散度。
    
    Args:
        model: 已訓練的模型，用於提取特徵
        sample: 單個樣本數據
        dataset: 參考數據集
        method: 密度計算方法
        k: 如使用kNN，指定k值
        
    Returns:
        float: 密度分數，較低值表示樣本較為孤立
    """
    # Ensure sample is a tensor with batch dimension
    if not isinstance(sample, torch.Tensor):
        sample = torch.tensor(sample, dtype=torch.float32)
    
    if sample.dim() == 1:
        sample = sample.unsqueeze(0)
    
    # Extract features from all samples in the dataset
    features = _extract_features(model, dataset)
    
    # Extract feature for the target sample
    sample_feature = _extract_features(model, [(sample, torch.tensor(0))])[0]
    
    if method == 'knn':
        # Use k-nearest neighbors to calculate density
        nn = NearestNeighbors(n_neighbors=min(k+1, len(features)))
        nn.fit(features)
        
        # Compute distances to k nearest neighbors
        distances, _ = nn.kneighbors([sample_feature])
        
        # Average distance to k nearest neighbors (excluding self)
        avg_distance = np.mean(distances[0][1:]) if len(distances[0]) > 1 else 0
        
        # Convert to density score (inversely proportional to distance)
        # Normalize to [0, 1] range
        density = 1.0 / (1.0 + avg_distance)
        
    elif method == 'relative_density':
        # Calculate average pairwise distance in feature space
        all_features = np.vstack([features, [sample_feature]])
        n_samples = len(all_features)
        
        # Compute a subset of distances for efficiency
        max_samples = min(100, n_samples)
        indices = np.random.choice(n_samples, max_samples, replace=False)
        subset_features = all_features[indices]
        
        # Compute pairwise distances for the subset
        nn = NearestNeighbors(n_neighbors=min(k+1, len(subset_features)))
        nn.fit(subset_features)
        
        # Get average distances for each point in subset
        all_avg_distances = []
        for i in range(len(subset_features)):
            distances, _ = nn.kneighbors([subset_features[i]])
            all_avg_distances.append(np.mean(distances[0][1:]))
        
        # Overall average distance in the feature space
        global_avg_distance = np.mean(all_avg_distances)
        
        # Get distance for the sample
        nn = NearestNeighbors(n_neighbors=min(k+1, len(features)))
        nn.fit(features)
        distances, _ = nn.kneighbors([sample_feature])
        sample_avg_distance = np.mean(distances[0][1:])
        
        # Relative density (ratio of sample's density to average density)
        # Lower value means the sample is more isolated
        if global_avg_distance > 0:
            density = global_avg_distance / (sample_avg_distance + 1e-10)
        else:
            density = 0.0
        
        # Normalize to [0, 1] range
        density = min(density, 1.0)
    
    else:
        raise ValueError(f"Unsupported density calculation method: {method}")
    
    return density

def _extract_features(model, dataset):
    """Extract features from a dataset using a model."""
    if isinstance(dataset, list):
        # If dataset is a list of (sample, target) tuples
        samples = [item[0] for item in dataset]
        
        # Convert to tensors if they are not
        if not isinstance(samples[0], torch.Tensor):
            samples = [torch.tensor(s, dtype=torch.float32) for s in samples]
        
        # Add batch dimension if needed
        samples = [s.unsqueeze(0) if s.dim() == 1 else s for s in samples]
        
        # Concatenate into a batch
        batch = torch.cat(samples, dim=0)
    else:
        # If dataset is a full Dataset object, sample a subset for efficiency
        max_samples = min(100, len(dataset))
        indices = np.random.choice(len(dataset), max_samples, replace=False)
        batch = torch.stack([dataset[i][0] for i in indices])
    
    # Get device
    device = next(model.parameters()).device
    
    # Move data to device
    batch = batch.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Extract features
    features = []
    with torch.no_grad():
        # For CNN models that follow a typical pattern
        if hasattr(model, 'features') and hasattr(model, 'classifier'):
            # Process in smaller batches if needed
            batch_size = 32
            for i in range(0, len(batch), batch_size):
                batch_part = batch[i:i+batch_size]
                x = model.features(batch_part)
                if hasattr(model, 'adaptive_pool'):
                    x = model.adaptive_pool(x)
                x = x.view(x.size(0), -1)
                features.append(x.cpu().numpy())
        else:
            # Use model outputs as features
            for i in range(0, len(batch), 32):
                batch_part = batch[i:i+32]
                outputs = model(batch_part)
                if outputs.dim() > 1:
                    # For classification, use pre-softmax outputs
                    features.append(outputs.cpu().numpy())
                else:
                    # For regression/ranking, reshape to 2D
                    features.append(outputs.view(-1, 1).cpu().numpy())
    
    # Combine batches
    features = np.vstack(features)
    
    return features

def calculate_prediction_stability(model, sample, augmentations=None, n_augmentations=10):
    """計算模型對樣本預測的穩定性。
    
    Args:
        model: 已訓練的模型
        sample: 單個樣本數據
        augmentations: 增強變換函數列表
        n_augmentations: 應用增強的次數
        
    Returns:
        float: 預測穩定性分數，越高表示越穩定
    """
    # Ensure sample is a tensor with batch dimension
    if not isinstance(sample, torch.Tensor):
        sample = torch.tensor(sample, dtype=torch.float32)
    
    if sample.dim() == 1:
        sample = sample.unsqueeze(0)
    
    # Get device
    device = next(model.parameters()).device
    
    # Move sample to device
    sample = sample.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # If no augmentations provided, use simple noise augmentation
    if augmentations is None:
        def add_noise(x, noise_level=0.05):
            return x + torch.randn_like(x) * noise_level
        
        augmentations = [
            lambda x: add_noise(x, 0.01),
            lambda x: add_noise(x, 0.05),
            lambda x: add_noise(x, 0.1)
        ]
    
    # Get original prediction
    with torch.no_grad():
        original_output = model(sample)
    
    # Store all predictions
    all_predictions = []
    
    # Apply augmentations and collect predictions
    for _ in range(n_augmentations):
        # Randomly select an augmentation
        aug_idx = np.random.randint(0, len(augmentations))
        augment_fn = augmentations[aug_idx]
        
        # Apply augmentation
        aug_sample = augment_fn(sample.clone())
        
        # Get prediction
        with torch.no_grad():
            aug_output = model(aug_sample)
        
        all_predictions.append(aug_output.cpu().numpy())
    
    # Calculate stability based on prediction variance
    if original_output.dim() > 1 and original_output.shape[1] > 1:
        # For classification tasks
        # Convert outputs to probabilities
        probs = [torch.nn.functional.softmax(torch.tensor(pred), dim=1).numpy() 
                for pred in all_predictions]
        
        # Calculate standard deviation of class probabilities
        probs_std = np.std(probs, axis=0)[0]
        
        # Average standard deviation across all classes
        avg_std = np.mean(probs_std)
        
        # Stability is inversely proportional to standard deviation
        stability = 1.0 - min(avg_std * 5, 1.0)  # Scale and limit to [0, 1]
    else:
        # For regression/ranking tasks
        preds = np.vstack(all_predictions)
        
        # Normalize predictions
        preds_mean = np.mean(preds)
        preds_range = np.max(preds) - np.min(preds) if np.max(preds) > np.min(preds) else 1.0
        
        # Calculate coefficient of variation (normalized standard deviation)
        if preds_range > 0:
            normalized_std = np.std(preds) / preds_range
        else:
            normalized_std = 0
        
        # Stability is inversely proportional to normalized standard deviation
        stability = 1.0 - min(normalized_std * 5, 1.0)  # Scale and limit to [0, 1]
    
    return stability

def calculate_loss_landscape(model, sample, target, epsilon=0.1, n_points=10):
    """分析樣本在損失景觀中的位置特性。
    
    Args:
        model: 已訓練的模型
        sample: 單個樣本數據
        target: 樣本標籤
        epsilon: 擾動範圍
        n_points: 每個維度的採樣點數
        
    Returns:
        dict: 損失景觀分析結果
    """
    # Ensure sample is a tensor with batch dimension
    if not isinstance(sample, torch.Tensor):
        sample = torch.tensor(sample, dtype=torch.float32)
    
    if sample.dim() == 1:
        sample = sample.unsqueeze(0)
    
    # Ensure target is a tensor
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target)
    
    if target.dim() == 0:
        target = target.unsqueeze(0)
    
    # Get device
    device = next(model.parameters()).device
    
    # Move data to device
    sample = sample.to(device)
    target = target.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Calculate original loss
    with torch.no_grad():
        original_output = model(sample)
        
        # Handle different output shapes
        if original_output.dim() == 1 or original_output.shape[1] == 1:
            # For regression or ranking tasks
            original_output = original_output.view(-1)
            if target.dim() > 1:
                target = target.view(-1)
            original_loss = torch.nn.functional.mse_loss(original_output, target.float())
        else:
            # For classification tasks
            if target.dim() > 1:
                # If target is one-hot encoded
                target = target.argmax(dim=1)
            original_loss = torch.nn.functional.cross_entropy(original_output, target)
    
    original_loss = original_loss.item()
    
    # Generate perturbation directions
    # For simplicity, we'll use two random directions
    random_dirs = []
    for _ in range(2):  # Two random directions
        random_dir = torch.randn_like(sample)
        # Normalize
        random_dir = random_dir / torch.norm(random_dir)
        random_dirs.append(random_dir)
    
    # Sample points in loss landscape
    alphas = np.linspace(-epsilon, epsilon, n_points)
    loss_values = np.zeros((n_points, n_points))
    
    for i, alpha1 in enumerate(alphas):
        for j, alpha2 in enumerate(alphas):
            # Generate perturbed sample
            perturbed_sample = sample + alpha1 * random_dirs[0] + alpha2 * random_dirs[1]
            
            # Compute loss for perturbed sample
            with torch.no_grad():
                perturbed_output = model(perturbed_sample)
                
                # Handle different output shapes
                if perturbed_output.dim() == 1 or perturbed_output.shape[1] == 1:
                    perturbed_output = perturbed_output.view(-1)
                    perturbed_loss = torch.nn.functional.mse_loss(perturbed_output, target.float())
                else:
                    perturbed_loss = torch.nn.functional.cross_entropy(perturbed_output, target)
            
            loss_values[i, j] = perturbed_loss.item()
    
    # Calculate curvature (approximation using second derivatives)
    # Compute numerical second derivatives in both directions
    d2l_dalpha1_2 = np.zeros(n_points)
    for i in range(1, n_points-1):
        d2l_dalpha1_2[i] = (loss_values[i+1, n_points//2] - 2*loss_values[i, n_points//2] + loss_values[i-1, n_points//2]) / ((alphas[1] - alphas[0])**2)
    
    d2l_dalpha2_2 = np.zeros(n_points)
    for j in range(1, n_points-1):
        d2l_dalpha2_2[j] = (loss_values[n_points//2, j+1] - 2*loss_values[n_points//2, j] + loss_values[n_points//2, j-1]) / ((alphas[1] - alphas[0])**2)
    
    # Average curvature
    avg_curvature = (np.mean(d2l_dalpha1_2[1:-1]) + np.mean(d2l_dalpha2_2[1:-1])) / 2.0
    
    # Sharp curvature may indicate problematic samples
    return {
        'loss_values': loss_values.tolist(),
        'curvature': float(avg_curvature),
        'original_loss': original_loss,
        'min_loss': float(np.min(loss_values)),
        'max_loss': float(np.max(loss_values)),
        'loss_range': float(np.max(loss_values) - np.min(loss_values))
    }

def calculate_comprehensive_quality_score(metrics_dict, weights=None):
    """基於多個指標計算綜合質量分數。
    
    Args:
        metrics_dict: 包含各項指標的字典
        weights: 各指標的權重
        
    Returns:
        float: 綜合質量分數
    """
    # Default weights if not provided
    if weights is None:
        weights = {}
        for key in metrics_dict:
            weights[key] = 1.0
    
    # Ensure weights for all metrics
    for key in metrics_dict:
        if key not in weights:
            weights[key] = 1.0
    
    # Calculate weighted average
    total_weight = sum(weights.values())
    
    if total_weight == 0:
        return 0.0
    
    weighted_sum = 0.0
    for key, value in metrics_dict.items():
        if key in weights:
            weighted_sum += value * weights[key]
    
    return weighted_sum / total_weight 