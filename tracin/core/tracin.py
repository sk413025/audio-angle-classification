"""
TracIn implementation for calculating the influence of training data examples.

Based on "Estimating Training Data Influence by Tracing Gradient Descent"
by Pang Wei Koh and Percy Liang.

This module allows the calculation of influence scores without modifying 
the existing training pipeline. It hooks into the checkpoints saved during
training to compute gradients and influence scores.
"""

import os
import json
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Callable
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


def get_default_device():
    """
    Get the default device for tensor operations.
    Returns CUDA if available, then MPS if available, otherwise CPU.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


class TracInCP:
    """
    TracIn with checkpoints implementation.
    
    This class implements the TracIn method using model checkpoints saved during training.
    It computes the influence of training examples on test examples by tracing the
    optimization trajectory using saved checkpoints.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        checkpoints: List[str],
        loss_fn: Callable,
        device: torch.device = None
    ):
        """
        Initialize TracInCP.
        
        Args:
            model: The model architecture (un-trained instance)
            checkpoints: List of file paths to the saved model checkpoints
            loss_fn: Loss function used in training (e.g., nn.CrossEntropyLoss())
            device: Device to perform computations on
        """
        # Use provided device or get default device
        if device is None:
            device = get_default_device()
            
        self.model = model
        self.checkpoints = checkpoints
        self.loss_fn = loss_fn
        self.device = device
        
        # Validate checkpoints
        for checkpoint in self.checkpoints:
            if not os.path.exists(checkpoint):
                raise FileNotFoundError(f"Checkpoint {checkpoint} does not exist")
                
        print(f"TracIn initialized with {len(checkpoints)} checkpoints on {device}")
    
    def compute_gradients(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor,
        checkpoint: str
    ) -> List[torch.Tensor]:
        """
        Compute gradients of the loss with respect to model parameters.
        
        Args:
            inputs: Input tensor
            targets: Target tensor
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
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Forward pass
        outputs = self.model(inputs)
        
        # Compute loss
        loss = self.loss_fn(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Get gradients
        gradients = []
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                gradients.append(param.grad.clone().detach())
                
        return gradients
    
    def compute_influence(
        self,
        train_dataset: Dataset,
        test_inputs: torch.Tensor,
        test_targets: torch.Tensor,
        batch_size: int = 32,
        num_workers: int = 4
    ) -> Dict[Union[int, str], float]:
        """
        Compute the influence scores of all training examples on a test example.
        
        Args:
            train_dataset: Dataset containing training examples
            test_inputs: Input tensor for the test example
            test_targets: Target tensor for the test example
            batch_size: Batch size for processing training examples
            num_workers: Number of workers for the DataLoader
            
        Returns:
            Dictionary mapping training example indices to influence scores
        """
        # Create dataloader for training examples
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        # Initialize influence scores
        influence_scores = {}
        
        # Compute gradients for the test example
        test_gradients_per_checkpoint = []
        for checkpoint in self.checkpoints:
            test_gradients = self.compute_gradients(test_inputs, test_targets, checkpoint)
            test_gradients_per_checkpoint.append(test_gradients)
        
        # Process training examples in batches
        for batch_idx, (train_inputs, train_targets, indices) in enumerate(train_loader):
            batch_influence = np.zeros(len(indices))
            
            # Compute influence across all checkpoints
            for checkpoint_idx, checkpoint in enumerate(self.checkpoints):
                train_gradients = self.compute_gradients(train_inputs, train_targets, checkpoint)
                test_gradients = test_gradients_per_checkpoint[checkpoint_idx]
                
                # Compute dot product between train and test gradients
                batch_dot_product = 0
                for train_grad, test_grad in zip(train_gradients, test_gradients):
                    # Flatten the gradients and compute dot product
                    batch_dot_product += torch.sum(
                        train_grad.flatten(1) * test_grad.flatten(1), dim=1
                    ).cpu().numpy()
                
                # Add to influence score
                batch_influence += batch_dot_product
                
            # Save influence scores
            if isinstance(indices, torch.Tensor) and indices.dtype == torch.int64:
                # If indices are integers
                for i, idx in enumerate(indices.cpu().numpy()):
                    influence_scores[int(idx)] = float(batch_influence[i])
            else:
                # For string or other types of indices
                for i, idx in enumerate(indices):
                    influence_scores[idx] = float(batch_influence[i])
            
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx+1}/{len(train_loader)} batches")
        
        return influence_scores

    def compute_self_influence(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 4
    ) -> Dict[Union[int, str], float]:
        """
        Compute the self-influence scores of all examples in a dataset.
        Self-influence is a measure of the difficulty of an example.
        
        Args:
            dataset: Dataset containing examples
            batch_size: Batch size for processing examples
            num_workers: Number of workers for the DataLoader
            
        Returns:
            Dictionary mapping example indices to self-influence scores
        """
        # Create dataloader for examples
        loader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        # Initialize self-influence scores
        self_influence_scores = {}
        
        # Process examples in batches
        for batch_idx, (inputs, targets, indices) in enumerate(loader):
            batch_self_influence = np.zeros(len(indices))
            
            # Compute self-influence across all checkpoints
            for checkpoint in self.checkpoints:
                # Compute gradients
                gradients = self.compute_gradients(inputs, targets, checkpoint)
                
                # Compute gradient norms
                batch_grad_norm_squared = 0
                for grad in gradients:
                    # Check gradient dimensions
                    if grad.dim() <= 1:
                        # If gradient is 1D, just square and sum
                        batch_grad_norm_squared += (grad ** 2).sum().cpu().numpy()
                    else:
                        # If gradient is multi-dimensional, flatten each sample
                        # and then compute squared norm
                        batch_grad_norm_squared += torch.sum(
                            grad.reshape(grad.size(0), -1) ** 2, dim=1
                        ).cpu().numpy()
                
                # Add to self-influence score
                batch_self_influence += batch_grad_norm_squared
                
            # Save self-influence scores
            if isinstance(indices, torch.Tensor) and indices.dtype == torch.int64:
                # If indices are integers
                for i, idx in enumerate(indices.cpu().numpy()):
                    self_influence_scores[int(idx)] = float(batch_self_influence[i])
            else:
                # For string or other types of indices
                for i, idx in enumerate(indices):
                    self_influence_scores[idx] = float(batch_self_influence[i])
            
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx+1}/{len(loader)} batches")
        
        return self_influence_scores

    def save_influence_scores(
        self,
        influence_scores: Dict[Union[int, str], float],
        metadata_path: str,
        score_name: str = "tracin_influence"
    ) -> None:
        """
        Save the influence scores to a metadata file.
        
        Args:
            influence_scores: Dictionary mapping example indices to influence scores
            metadata_path: Path to save the metadata
            score_name: Name of the score in the metadata file
        """
        # Make directory if it doesn't exist
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        # Load existing metadata if it exists
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # Update metadata with influence scores
        for idx, score in influence_scores.items():
            idx_str = str(idx)  # Convert any index to string for JSON
            if idx_str not in metadata:
                metadata[idx_str] = {}
            metadata[idx_str][score_name] = score
        
        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Saved {len(influence_scores)} influence scores to {metadata_path}") 