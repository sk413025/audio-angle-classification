"""
TracIn implementation specifically adapted for the ranking task.

This module extends the base TracIn implementation to work with
our specific ranking datasets and loss functions.
"""

import os
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset

from utils.tracin.tracin import TracInCP, get_default_device
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
    ) -> Dict[str, float]:
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
            Dictionary mapping pair indices to influence scores
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
                    
                    # Add to influence score
                    batch_influence += current_dot_products
                    
                except Exception as e:
                    print(f"Error processing checkpoint {checkpoint}: {e}")
                    # Continue with the next checkpoint
                    continue
            
            # Save influence scores
            for i, idx in enumerate(indices):
                influence_scores[idx] = float(batch_influence[i])
            
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx+1}/{len(dataloader)} batches")
        
        return influence_scores
    
    def compute_self_influence_for_pairs(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 4
    ) -> Dict[str, float]:
        """
        Compute the self-influence scores of all pairs in a dataset.
        
        Args:
            dataset: Dataset containing ranking pairs
            batch_size: Batch size for processing examples
            num_workers: Number of workers for the DataLoader
            
        Returns:
            Dictionary mapping pair indices to self-influence scores
        """
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        # Initialize self-influence scores
        self_influence_scores = {}
        
        # Process pairs in batches
        for batch_idx, batch_data in enumerate(dataloader):
            # Extract batch data - RankingPairDataset returns 7 values
            x1_batch, x2_batch, targets_batch, _, _, sample_id1_batch, sample_id2_batch = batch_data
            
            # Use sample IDs as indices for the influence scores
            indices = [f"{id1}_{id2}" for id1, id2 in zip(sample_id1_batch, sample_id2_batch)]
            current_batch_size = len(indices)
            
            batch_self_influence = np.zeros(current_batch_size)
            
            # Compute self-influence across all checkpoints
            for checkpoint in self.checkpoints:
                try:
                    # Compute gradients
                    gradients = self.compute_gradients_for_pair(
                        x1_batch, x2_batch, targets_batch, checkpoint
                    )
                    
                    # Compute gradient norms
                    current_grad_norm_squared = np.zeros(current_batch_size)
                    
                    for grad in gradients:
                        # Ensure we're handling each gradient tensor correctly based on its dimensions
                        if grad.dim() == 0:  # Scalar
                            norm_squared = (grad.item() ** 2)
                            current_grad_norm_squared += np.full(current_batch_size, norm_squared)
                        elif grad.dim() == 1:  # Vector
                            if grad.size(0) == current_batch_size:
                                # Gradient has one value per sample
                                norms = (grad ** 2).cpu().numpy()
                                current_grad_norm_squared += norms
                            else:
                                # Gradient is shared across samples
                                norm_squared = (grad ** 2).sum().item()
                                current_grad_norm_squared += np.full(current_batch_size, norm_squared / current_batch_size)
                        else:  # Multi-dimensional tensor
                            if grad.size(0) == current_batch_size:
                                # First dimension matches batch size, process each sample separately
                                sample_grads = grad.reshape(current_batch_size, -1)
                                norms = torch.sum(sample_grads ** 2, dim=1).cpu().numpy()
                                current_grad_norm_squared += norms
                            else:
                                # Reshape might not match batch size, use a more robust approach
                                total_norm_squared = torch.sum(grad ** 2).item()
                                current_grad_norm_squared += np.full(current_batch_size, total_norm_squared / current_batch_size)
                    
                    # Add to self-influence score
                    batch_self_influence += current_grad_norm_squared
                    
                except Exception as e:
                    print(f"Error processing checkpoint {checkpoint}: {e}")
                    # Continue with the next checkpoint
                    continue
            
            # Save self-influence scores
            for i, idx in enumerate(indices):
                self_influence_scores[idx] = float(batch_self_influence[i])
            
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx+1}/{len(dataloader)} batches")
        
        return self_influence_scores
    
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
            influence_scores: Dictionary mapping example indices to influence scores
            metadata_dir: Base metadata directory path
            material: Material type (e.g., "metal")
            frequency: Frequency type (e.g., "500hz")
            score_name: Name of the score in the metadata file
        """
        # Construct the metadata path following the project's convention
        metadata_path = os.path.join(
            metadata_dir,
            f"{material}_{frequency}_metadata.json"
        )
        
        # Use the parent class's save method
        self.save_influence_scores(influence_scores, metadata_path, score_name) 