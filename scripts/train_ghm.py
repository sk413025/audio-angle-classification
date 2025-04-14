#!/usr/bin/env python
"""
GHM Training Script

This script trains models using the GHM (Gradient Harmonizing Mechanism) loss function.

Usage:
    python scripts/train_ghm.py --frequency <freq> --bins <num_bins> --alpha <alpha>
    
Example:
    python scripts/train_ghm.py --frequency 1000hz --bins 30 --alpha 0.75
"""

import os
import argparse
import sys
import torch

# Add the root directory to the Python path to ensure imports work correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.training.ghm_trainer import train_with_ghm
from utils.data.dataset import get_dataloaders

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train models using GHM loss")
    
    parser.add_argument('--frequency', type=str, required=True,
                        choices=['500hz', '1000hz', '3000hz'],
                        help='Frequency data to use for training')
    
    parser.add_argument('--bins', type=int, default=30,
                        help='Number of bins for GHM loss (default: 30)')
    
    parser.add_argument('--alpha', type=float, default=0.75,
                        help='Alpha parameter for GHM loss (default: 0.75)')
    
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs (default: 150)')
    
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training and validation (default: 32)')
    
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    
    parser.add_argument('--model-type', type=str, default='resnet',
                        choices=['resnet', 'cnn', 'transformer'],
                        help='Model architecture (default: resnet)')
    
    parser.add_argument('--output-dir', type=str, default='saved_models',
                        help='Directory to save trained models')
    
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory to save training logs')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use for training (default: cuda if available, else cpu)')
    
    return parser.parse_args()

def main():
    """Main function for GHM training."""
    args = parse_arguments()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Determine the device to use
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Training on {device}")
    print(f"Frequency: {args.frequency}, Bins: {args.bins}, Alpha: {args.alpha}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create specific log directory for this run
    run_name = f"ghm_{args.frequency}_bins{args.bins}_alpha{args.alpha}"
    log_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Get dataloaders
    dataloaders = get_dataloaders(
        frequency=args.frequency,
        batch_size=args.batch_size,
        seed=args.seed
    )
    
    # Train the model
    model_path = train_with_ghm(
        dataloaders=dataloaders,
        bins=args.bins,
        alpha=args.alpha,
        model_type=args.model_type,
        frequency=args.frequency,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        output_dir=args.output_dir,
        log_dir=log_dir,
        seed=args.seed
    )
    
    print(f"Training complete. Model saved to {model_path}")

if __name__ == "__main__":
    main()