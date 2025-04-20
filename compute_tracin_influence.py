"""
Compute TracIn influence scores for training data.

This script:
1. Loads checkpoints saved during training
2. Computes self-influence scores for all training examples (measure of difficulty)
3. Optionally computes influence scores for specific test examples
4. Saves the scores to the project's metadata directory
"""

import os
import argparse
import torch
import glob
from pathlib import Path
import numpy as np
import json
import random
from torch.utils.data import DataLoader, random_split, Subset

import config
from models.resnet_ranker import SimpleCNNAudioRanker
from datasets import (
    AudioSpectrumDataset,
    RankingPairDataset,
    GHMAwareRankingDataset,
    DatasetConfig
)
from utils.tracin.ranking_tracin import RankingTracInCP
from utils.tracin.tracin import get_default_device
from utils.common_utils import set_seed


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compute TracIn influence scores")
    
    # Required arguments
    parser.add_argument('--frequency', type=str, required=True, 
                        choices=['500hz', '1000hz', '3000hz', 'all'],
                        help='Frequency data to use.')
    
    parser.add_argument('--material', type=str, default=config.MATERIAL,
                        help=f'Material type (default: {config.MATERIAL} from config.py).')
    
    # Model checkpoint arguments
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                        help='Directory containing model checkpoints.')
    
    parser.add_argument('--checkpoint-prefix', type=str, default="model_epoch_",
                        help='Prefix of checkpoint files to use.')
    
    # Score saving arguments
    parser.add_argument('--metadata-dir', type=str, 
                        default="/Users/sbplab/Hank/sinetone_sliced/step_018_sliced/metadata",
                        help='Directory to save metadata with influence scores.')
    
    # TracIn parameters
    parser.add_argument('--loss-type', type=str, default='standard',
                        choices=['standard', 'ghm'],
                        help='Type of loss function used during training (default: standard).')
    
    parser.add_argument('--margin', type=float, default=1.0,
                        help='Margin for the ranking loss.')
    
    parser.add_argument('--ghm-bins', type=int, default=10,
                        help='Number of bins for GHM loss (only used if loss-type is ghm).')
    
    parser.add_argument('--ghm-alpha', type=float, default=0.75,
                        help='Alpha parameter for GHM loss (only used if loss-type is ghm).')
    
    # Computation parameters
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for computing gradients.')
    
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers for data loading.')
    
    parser.add_argument('--dataset-version', type=str, default='1.0',
                        help='Dataset version to use.')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    
    parser.add_argument('--compute-self-influence', action='store_true',
                        help='Compute self-influence scores (measure of example difficulty).')
    
    # New parameters for influence calculation
    parser.add_argument('--compute-influence', action='store_true',
                        help='Compute influence scores of training examples on test examples.')
    
    parser.add_argument('--num-test-samples', type=int, default=5,
                        help='Number of test samples to compute influence for.')
    
    parser.add_argument('--influence-score-name', type=str, default='tracin_influence',
                        help='Name for the influence score in metadata file.')
    
    # Device selection
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], 
                        default=None,
                        help='Device to use for computation (default: auto-detect).')
    
    return parser.parse_args()


def find_checkpoints(checkpoint_dir, checkpoint_prefix):
    """Find checkpoint files matching the given prefix."""
    pattern = os.path.join(checkpoint_dir, f"{checkpoint_prefix}*.pt")
    checkpoints = sorted(glob.glob(pattern))
    
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found matching {pattern}")
    
    return checkpoints


def load_datasets(args):
    """Load datasets for computing influence scores."""
    print(f"Loading Dataset for {args.frequency} with material {args.material}")
    
    # Create dataset config
    dataset_config = DatasetConfig(
        version=args.dataset_version,
        description=f"TracIn computation for {args.material}_{args.frequency}"
    )
    
    # Load base dataset
    dataset = AudioSpectrumDataset(
        data_root=config.DATA_ROOT,
        classes=config.CLASSES,
        selected_seqs=config.SEQ_NUMS,
        selected_freq=args.frequency,
        material=args.material,
        exclusion_file=dataset_config.get_exclusion_path(),
        metadata_file=dataset_config.get_metadata_path()
    )
    
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Cannot compute influence scores.")
    
    # Split dataset into train and validation sets
    train_size = int(0.70 * len(dataset))
    val_size = len(dataset) - train_size
    
    if train_size < 4 or val_size < 4:
        raise ValueError(f"Dataset too small (Total: {len(dataset)}) for train/validation split.")
    
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    # Create ranking datasets
    train_ranking_dataset = RankingPairDataset(train_dataset)
    val_ranking_dataset = RankingPairDataset(val_dataset)
    
    print(f"Train examples: {len(train_dataset)}")
    print(f"Train ranking pairs: {len(train_ranking_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    print(f"Validation ranking pairs: {len(val_ranking_dataset)}")
    
    return {
        'train': train_dataset,
        'val': val_dataset,
        'train_ranking': train_ranking_dataset,
        'val_ranking': val_ranking_dataset,
        'raw': dataset
    }


def initialize_model_and_tracin(args, n_freqs):
    """Initialize model and TracIn module."""
    # Initialize model
    model = SimpleCNNAudioRanker(n_freqs=n_freqs)
    
    # Find checkpoints
    checkpoints = find_checkpoints(args.checkpoint_dir, args.checkpoint_prefix)
    print(f"Found {len(checkpoints)} checkpoints")
    
    # Set device based on args or auto-detect
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_default_device()
    
    print(f"Using device: {device}")
    
    # Initialize TracIn
    tracin = RankingTracInCP(
        model=model,
        checkpoints=checkpoints,
        loss_type=args.loss_type,
        margin=args.margin,
        ghm_bins=args.ghm_bins,
        ghm_alpha=args.ghm_alpha,
        device=device
    )
    
    return tracin


def compute_self_influence(args, tracin, datasets):
    """Compute self-influence scores for training examples."""
    print("Computing self-influence scores for training pairs...")
    
    # Compute self-influence for training pairs
    self_influence_scores = tracin.compute_self_influence_for_pairs(
        dataset=datasets['train_ranking'],
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Save self-influence scores
    tracin.save_to_project_metadata(
        influence_scores=self_influence_scores,
        metadata_dir=args.metadata_dir,
        material=args.material,
        frequency=args.frequency,
        score_name="tracin_self_influence"
    )
    
    print(f"Saved self-influence scores for {len(self_influence_scores)} training pairs")
    
    return self_influence_scores


def select_test_samples(args, datasets, self_influence_scores=None):
    """
    Select test samples for influence computation based on different strategies.
    
    If self_influence_scores are provided, select samples from different difficulty levels.
    Otherwise, select random samples from different classes.
    """
    val_dataset = datasets['val_ranking']
    num_samples = min(args.num_test_samples, len(val_dataset))
    
    if num_samples == 0:
        raise ValueError("No validation samples available for influence computation")
    
    # Strategy 1: If we have self-influence scores, select samples from different difficulty buckets
    if self_influence_scores:
        # We don't have self-influence for validation samples, so we'll use a random strategy
        print("Using random sampling from validation set")
        indices = random.sample(range(len(val_dataset)), num_samples)
        return [get_sample_from_dataset(val_dataset, idx) for idx in indices]
    
    # Strategy 2: Select samples randomly from different angle classes if possible
    print(f"Selecting {num_samples} test samples randomly from validation set")
    indices = random.sample(range(len(val_dataset)), num_samples)
    return [get_sample_from_dataset(val_dataset, idx) for idx in indices]


def get_sample_from_dataset(dataset, index):
    """Extract a single sample from the dataset at the given index."""
    sample_data = dataset[index]
    
    # For RankingPairDataset, we expect the format:
    # (x1, x2, target, y1, y2, id1, id2)
    x1, x2, target, _, _, id1, id2 = sample_data
    
    # Create a unique identifier for this test sample
    sample_id = f"{id1}_{id2}"
    
    return {
        'x1': x1.unsqueeze(0),  # Add batch dimension
        'x2': x2.unsqueeze(0),  # Add batch dimension
        'target': target.unsqueeze(0),  # Add batch dimension
        'id': sample_id
    }


def compute_influence(args, tracin, datasets, test_samples):
    """
    Compute influence scores of training examples on selected test examples.
    
    Args:
        args: Command line arguments
        tracin: Initialized TracIn module
        datasets: Dictionary of datasets
        test_samples: List of test samples to compute influence for
    """
    print(f"Computing influence scores for {len(test_samples)} test samples...")
    
    all_influence_scores = {}
    
    for i, test_sample in enumerate(test_samples):
        print(f"Computing influence for test sample {i+1}/{len(test_samples)}: {test_sample['id']}")
        
        # Compute influence scores for this test sample
        influence_scores = tracin.compute_influence_for_pair(
            dataset=datasets['train_ranking'],
            test_x1=test_sample['x1'],
            test_x2=test_sample['x2'],
            test_target=test_sample['target'],
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # Add to combined influence scores dictionary
        for train_id, score in influence_scores.items():
            if train_id not in all_influence_scores:
                all_influence_scores[train_id] = {}
            
            all_influence_scores[train_id][f"{args.influence_score_name}_{test_sample['id']}"] = score
        
        # Save individual influence scores for this test sample
        individual_scores = {k: {f"{args.influence_score_name}": v} for k, v in influence_scores.items()}
        tracin.save_to_project_metadata(
            influence_scores=individual_scores,
            metadata_dir=args.metadata_dir,
            material=args.material,
            frequency=args.frequency,
            score_name=f"{args.influence_score_name}_{test_sample['id']}"
        )
        
        print(f"Saved influence scores for test sample {test_sample['id']}")
    
    # Save combined influence scores
    metadata_path = os.path.join(
        args.metadata_dir,
        f"{args.material}_{args.frequency}_influence_metadata.json"
    )
    
    # Make directory if it doesn't exist
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    
    # Load existing metadata if it exists
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    # Update metadata with influence scores
    for train_id, scores in all_influence_scores.items():
        if train_id not in metadata:
            metadata[train_id] = {}
        
        for score_name, score in scores.items():
            metadata[train_id][score_name] = score
    
    # Save updated metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved combined influence scores to {metadata_path}")


def main():
    """Main function."""
    args = parse_arguments()
    set_seed(args.seed)
    
    print("=== TracIn Influence Computation ===")
    print(f"Frequency: {args.frequency}")
    print(f"Material: {args.material}")
    print(f"Loss type: {args.loss_type}")
    
    # Ensure metadata directory exists
    os.makedirs(args.metadata_dir, exist_ok=True)
    
    try:
        # Load datasets
        datasets = load_datasets(args)
        
        # Initialize model and TracIn
        tracin = initialize_model_and_tracin(args, datasets['raw'].data.shape[2])
        
        # Initialize variables
        self_influence_scores = None
        
        # Compute self-influence if requested
        if args.compute_self_influence:
            self_influence_scores = compute_self_influence(args, tracin, datasets)
        
        # Compute influence if requested
        if args.compute_influence:
            # Select test samples
            test_samples = select_test_samples(args, datasets, self_influence_scores)
            
            # Compute influence scores
            compute_influence(args, tracin, datasets, test_samples)
        
        print("TracIn computation completed successfully")
        
    except Exception as e:
        print(f"Error during TracIn computation: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main() 