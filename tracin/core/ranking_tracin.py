"""
TracIn implementation specifically adapted for the ranking task.

This module extends the base TracIn implementation to work with
our specific ranking datasets and loss functions.
"""

import os
import torch
import numpy as np
import json
from typing import List, Dict, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset

# 更新導入路徑以使用新的模組結構
from tracin.core.tracin import TracInCP, get_default_device
from models.resnet_ranker import SimpleCNNAudioRanker
from losses.ghm_loss import GHMRankingLoss
from torch.nn import MarginRankingLoss


class RankingTracInCP(TracInCP):
    """
    TracIn implementation adapted for our audio ranking task.
    
    This class handles the specifics of computing influence scores
    for ranking pairs, working with our project's datasets and models.
    """
    
    def __init__(
        self,
        model: SimpleCNNAudioRanker,
        checkpoints: List[str],
        loss_type: str = "standard",
        margin: float = 1.0,
        ghm_bins: int = 10,
        ghm_alpha: float = 0.75,
        device: torch.device = None
    ):
        """
        Initialize RankingTracInCP.
        
        Args:
            model: The model architecture (un-trained instance of SimpleCNNAudioRanker)
            checkpoints: List of file paths to the saved model checkpoints
            loss_type: Type of loss function ('standard' or 'ghm')
            margin: Margin parameter for the ranking loss
            ghm_bins: Number of bins for GHM loss
            ghm_alpha: Alpha parameter for GHM loss
            device: Device to perform computations on
        """
        # Get device if not provided
        if device is None:
            device = get_default_device()
            
        # Create loss function based on loss_type
        if loss_type == "ghm":
            loss_fn = GHMRankingLoss(
                margin=margin,
                bins=ghm_bins,
                alpha=ghm_alpha
            )
        else:  # "standard"
            loss_fn = MarginRankingLoss(margin=margin)
        
        # Initialize the parent class
        super().__init__(model, checkpoints, loss_fn, device)
        
        self.loss_type = loss_type
        print(f"RankingTracInCP initialized with {loss_type} loss on {device}")
    
    def compute_gradients_for_pair(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        target: torch.Tensor,
        checkpoint: str
    ) -> List[torch.Tensor]:
        """
        Compute gradients for a ranking pair.
        
        Args:
            x1: First input tensor in the pair
            x2: Second input tensor in the pair
            target: Target tensor (1 if x1 > x2, -1 if x1 < x2)
            checkpoint: Path to the model checkpoint to use
            
        Returns:
            List of gradient tensors for each parameter
        """
        # Load the checkpoint
        checkpoint_data = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint_data['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Move inputs to device
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)
        target = target.to(self.device)
        
        # Reshape target to match scores dimensions if needed
        # scores shape is typically [batch_size, 1] and target is [batch_size]
        if target.dim() == 1:
            target = target.unsqueeze(1)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Forward pass
        scores1 = self.model(x1)
        scores2 = self.model(x2)
        
        # Compute loss
        if self.loss_type == "ghm":
            # For GHM loss, we need to pass both scores directly
            loss = self.loss_fn(scores1, scores2, target)
        else:
            # For standard ranking loss
            loss = self.loss_fn(scores1, scores2, target)
        
        # Backward pass
        loss.backward()
        
        # Get gradients
        gradients = []
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                gradients.append(param.grad.clone().detach())
                
        return gradients
    
    def compute_influence_for_pair(
        self,
        dataset: Dataset,
        test_x1: torch.Tensor,
        test_x2: torch.Tensor,
        test_target: torch.Tensor,
        batch_size: int = 32,
        num_workers: int = 4
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Compute the influence scores of all pairs in a dataset on a test pair.
        
        Args:
            dataset: Dataset containing ranking pairs
            test_x1: First input tensor of the test pair
            test_x2: Second input tensor of the test pair
            test_target: Target tensor of the test pair
            batch_size: Batch size for processing examples
            num_workers: Number of workers for the DataLoader
            
        Returns:
            Tuple containing:
            1. Dictionary mapping pair indices to total influence scores
            2. Dictionary mapping pair indices to per-checkpoint influence scores
        """
        # Move test data to device
        test_x1 = test_x1.to(self.device)
        test_x2 = test_x2.to(self.device)
        test_target = test_target.to(self.device)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        # Initialize influence scores
        influence_scores = {}
        # New dictionary to store per-checkpoint influence scores
        per_checkpoint_influences = {}
        
        # Get checkpoint names (extract from paths)
        checkpoint_names = [os.path.basename(cp) for cp in self.checkpoints]
        
        # Compute gradients for the test pair
        test_gradients_per_checkpoint = []
        for checkpoint in self.checkpoints:
            test_gradients = self.compute_gradients_for_pair(
                test_x1, test_x2, test_target, checkpoint
            )
            test_gradients_per_checkpoint.append(test_gradients)
        
        # Process pairs in batches
        for batch_idx, batch_data in enumerate(dataloader):
            # Extract batch data - RankingPairDataset returns 7 values
            x1_batch, x2_batch, targets_batch, _, _, sample_id1_batch, sample_id2_batch = batch_data
            
            # Use sample IDs as indices for the influence scores
            indices = [f"{id1}_{id2}" for id1, id2 in zip(sample_id1_batch, sample_id2_batch)]
            current_batch_size = len(indices)
            
            batch_influence = np.zeros(current_batch_size)
            # New array to store per-checkpoint influence for this batch
            batch_per_checkpoint_influence = [np.zeros(current_batch_size) for _ in self.checkpoints]
            
            # Compute influence across all checkpoints
            for checkpoint_idx, checkpoint in enumerate(self.checkpoints):
                try:
                    # Compute gradients for each pair in the batch
                    train_gradients = self.compute_gradients_for_pair(
                        x1_batch, x2_batch, targets_batch, checkpoint
                    )
                    test_gradients = test_gradients_per_checkpoint[checkpoint_idx]
                    
                    # Compute dot product between train and test gradients
                    current_dot_products = np.zeros(current_batch_size)
                    
                    for train_grad, test_grad in zip(train_gradients, test_gradients):
                        # Handle different gradient shapes and dimensions
                        if train_grad.dim() == 0 or test_grad.dim() == 0:  # Scalar
                            # Simple multiplication for scalars
                            dot_product = (train_grad.item() * test_grad.item())
                            current_dot_products += np.full(current_batch_size, dot_product)
                        elif train_grad.dim() == 1 and test_grad.dim() == 1:  # Vector
                            if train_grad.size(0) == current_batch_size:
                                # Train gradient has one value per sample
                                if test_grad.size(0) == 1 or test_grad.size(0) == current_batch_size:
                                    # Test gradient is compatible
                                    test_expanded = test_grad.expand(current_batch_size) if test_grad.size(0) == 1 else test_grad
                                    dot_products = (train_grad * test_expanded).cpu().numpy()
                                    current_dot_products += dot_products
                                else:
                                    # Incompatible sizes, use scalar product
                                    dot_product = torch.dot(train_grad, test_grad).item() / current_batch_size
                                    current_dot_products += np.full(current_batch_size, dot_product)
                            else:
                                # Both are vectors but not per-sample, use regular dot product
                                dot_product = torch.dot(train_grad, test_grad).item() / current_batch_size
                                current_dot_products += np.full(current_batch_size, dot_product)
                        else:  # Multi-dimensional tensors
                            if train_grad.size(0) == current_batch_size:
                                # First dimension of train_grad matches batch size
                                train_flat = train_grad.reshape(current_batch_size, -1)
                                
                                if test_grad.dim() > 1 and test_grad.size(0) == current_batch_size:
                                    # Both have matching batch dimension
                                    test_flat = test_grad.reshape(current_batch_size, -1)
                                    # Compute per-sample dot product
                                    dot_products = torch.sum(train_flat * test_flat, dim=1).cpu().numpy()
                                    current_dot_products += dot_products
                                else:
                                    # Test gradient doesn't match batch size, flatten and distribute
                                    test_flat = test_grad.reshape(-1)
                                    # Compute dot product with each sample
                                    for i in range(current_batch_size):
                                        current_dot_products[i] += torch.dot(train_flat[i], test_flat).item()
                            else:
                                # Gradient shapes are incompatible, use a fallback approach
                                train_flat = train_grad.reshape(-1)
                                test_flat = test_grad.reshape(-1)
                                dot_product = torch.dot(train_flat, test_flat).item() / current_batch_size
                                current_dot_products += np.full(current_batch_size, dot_product)
                    
                    # Store per-checkpoint influence
                    batch_per_checkpoint_influence[checkpoint_idx] = current_dot_products
                    
                    # Add to total influence score
                    batch_influence += current_dot_products
                    
                except Exception as e:
                    print(f"Error computing influence for checkpoint {checkpoint}: {e}")
                    # Continue with the next checkpoint
            
            # Store influence scores
            for i, idx in enumerate(indices):
                influence_scores[idx] = float(batch_influence[i])
                
                # Store per-checkpoint influence scores
                if idx not in per_checkpoint_influences:
                    per_checkpoint_influences[idx] = {}
                
                for cp_idx, cp_name in enumerate(checkpoint_names):
                    per_checkpoint_influences[idx][cp_name] = float(batch_per_checkpoint_influence[cp_idx][i])
            
            if batch_idx % 10 == 0 or batch_idx + 1 == len(dataloader):
                print(f"Processed {batch_idx+1}/{len(dataloader)} batches")
        
        return influence_scores, per_checkpoint_influences
    
    def compute_self_influence_for_pairs(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 4
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Compute self-influence scores for all pairs in the dataset.
        Self-influence is a measure of the difficulty of a training example.
        
        Args:
            dataset: Dataset containing ranking pairs
            batch_size: Batch size for processing examples
            num_workers: Number of workers for the DataLoader
            
        Returns:
            Tuple containing:
            1. Dictionary mapping pair indices to total self-influence scores
            2. Dictionary mapping pair indices to per-checkpoint self-influence scores
        """
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        # Initialize influence scores
        self_influence_scores = {}
        # New dictionary to store per-checkpoint self-influence scores
        per_checkpoint_self_influences = {}
        
        # Get checkpoint names (extract from paths)
        checkpoint_names = [os.path.basename(cp) for cp in self.checkpoints]
        
        # Process pairs in batches
        for batch_idx, batch_data in enumerate(dataloader):
            # Extract batch data
            x1_batch, x2_batch, targets_batch, _, _, sample_id1_batch, sample_id2_batch = batch_data
            
            # Use sample IDs as indices for the influence scores
            indices = [f"{id1}_{id2}" for id1, id2 in zip(sample_id1_batch, sample_id2_batch)]
            current_batch_size = len(indices)
            
            # Initialize influence scores for this batch
            batch_self_influence = np.zeros(current_batch_size)
            # New array to store per-checkpoint self-influence for this batch
            batch_per_checkpoint_self_influence = [np.zeros(current_batch_size) for _ in self.checkpoints]
            
            # Compute self-influence across all checkpoints
            for checkpoint_idx, checkpoint in enumerate(self.checkpoints):
                try:
                    # Compute gradients for this batch
                    gradients = self.compute_gradients_for_pair(
                        x1_batch, x2_batch, targets_batch, checkpoint
                    )
                    
                    # Compute gradient norm squared for each example in batch
                    current_grad_norm_squared = np.zeros(current_batch_size)
                    
                    for grad in gradients:
                        if grad.size(0) == current_batch_size:
                            # Gradient has batch dimension
                            grad_flat = grad.reshape(current_batch_size, -1)
                            grad_norm_squared = torch.sum(grad_flat ** 2, dim=1).cpu().numpy()
                            current_grad_norm_squared += grad_norm_squared
                        else:
                            # Gradient does not have batch dimension or is scalar
                            grad_norm_squared = torch.sum(grad ** 2).item() / current_batch_size
                            current_grad_norm_squared += np.full(current_batch_size, grad_norm_squared)
                    
                    # Store per-checkpoint self-influence
                    batch_per_checkpoint_self_influence[checkpoint_idx] = current_grad_norm_squared
                    
                    # Add to total self-influence score
                    batch_self_influence += current_grad_norm_squared
                    
                except Exception as e:
                    print(f"Error computing self-influence for checkpoint {checkpoint}: {e}")
                    # Continue with the next checkpoint
            
            # Store self-influence scores
            for i, idx in enumerate(indices):
                self_influence_scores[idx] = float(batch_self_influence[i])
                
                # Store per-checkpoint self-influence scores
                if idx not in per_checkpoint_self_influences:
                    per_checkpoint_self_influences[idx] = {}
                
                for cp_idx, cp_name in enumerate(checkpoint_names):
                    per_checkpoint_self_influences[idx][cp_name] = float(batch_per_checkpoint_self_influence[cp_idx][i])
            
            if batch_idx % 10 == 0 or batch_idx + 1 == len(dataloader):
                print(f"Processed {batch_idx+1}/{len(dataloader)} batches")
        
        return self_influence_scores, per_checkpoint_self_influences
    
    def save_to_project_metadata(
        self,
        influence_scores: Dict[str, float],
        metadata_dir: str,
        material: str,
        frequency: str,
        score_name: str = "tracin_influence"
    ) -> None:
        """
        Save influence scores to the project's metadata directory.
        
        Args:
            influence_scores: Dictionary mapping training example indices to influence scores
            metadata_dir: Directory path to save the metadata
            material: Material type (e.g., 'metal', 'plastic')
            frequency: Frequency data (e.g., '500hz', '1000hz')
            score_name: Name for the influence score in metadata file
        """
        # Create path to metadata file
        metadata_path = os.path.join(
            metadata_dir,
            f"{material}_{frequency}_metadata.json"
        )
        
        # Make directory if it doesn't exist
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        # Load existing metadata if it exists
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # Update metadata with influence scores
        for idx, score in influence_scores.items():
            if idx not in metadata:
                metadata[idx] = {}
            
            # Add new score to metadata
            metadata[idx][score_name] = score
        
        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved {len(influence_scores)} influence scores to {metadata_path}")
    
    def save_per_checkpoint_influence(
        self,
        per_checkpoint_influences: Dict[str, Dict[str, float]],
        metadata_dir: str,
        material: str,
        frequency: str,
        score_name: str = "tracin_influence"
    ) -> None:
        """
        Save per-checkpoint influence scores to the project's metadata directory.
        
        Args:
            per_checkpoint_influences: Dictionary mapping training example indices to per-checkpoint influence scores
            metadata_dir: Directory path to save the metadata
            material: Material type (e.g., 'metal', 'plastic')
            frequency: Frequency data (e.g., '500hz', '1000hz')
            score_name: Base name for the influence score in metadata file
        """
        # Create path to metadata file
        metadata_path = os.path.join(
            metadata_dir,
            f"{material}_{frequency}_per_checkpoint_metadata.json"
        )
        
        # Make directory if it doesn't exist
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        # Load existing metadata if it exists
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # Update metadata with per-checkpoint influence scores
        for idx, checkpoint_scores in per_checkpoint_influences.items():
            if idx not in metadata:
                metadata[idx] = {}
            
            # Add new scores to metadata
            for checkpoint_name, score in checkpoint_scores.items():
                metadata[idx][f"{score_name}_{checkpoint_name}"] = score
        
        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved per-checkpoint influence scores for {len(per_checkpoint_influences)} examples to {metadata_path}") 