#!/usr/bin/env python
"""
SVRG Training Script

This script trains models using the SVRG (Stochastic Variance Reduced Gradient) optimizer.

Usage:
    python scripts/train_svrg.py --frequency <freq> --loss-type <loss_type>
    
Example:
    python scripts/train_svrg.py --frequency 1000hz --loss-type standard
"""

import os
import argparse
import sys
import torch
import pickle
from datetime import datetime
import platform

# Add the root directory to the Python path to ensure imports work correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import SpectrogramDatasetWithMaterial, RankingPairDataset
from models.resnet_ranker import SimpleCNNAudioRanker
from utils.svrg_optimizer import SVRG_k, SVRG_Snapshot
from torch.nn import MarginRankingLoss
from losses.ghm_loss import GHMRankingLoss
from utils.common_utils import worker_init_fn, set_seed
from utils.visualization import plot_training_history

import config

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train models using SVRG optimizer")
    
    parser.add_argument('--frequency', type=str, required=True,
                        choices=['500hz', '1000hz', '3000hz', 'all'],
                        help='Frequency data to use for training (or \'all\').')
    
    parser.add_argument('--material', type=str, default=config.MATERIAL,
                        help=f'Material type (default: {config.MATERIAL} from config.py).')
    
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs (default: 30).')
    
    parser.add_argument('--checkpoint-interval', type=int, default=5,
                        help='Save model checkpoint every N epochs (default: 5).')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42).')
    
    parser.add_argument('--loss-type', type=str, default='standard',
                        choices=['standard', 'ghm'],
                        help='Type of loss function to use (default: standard).')
    
    # GHM specific arguments
    parser.add_argument('--ghm-bins', type=int, default=10,
                        help='Number of bins for GHM loss (default: 10). Only used if --loss-type is ghm.')
    parser.add_argument('--ghm-alpha', type=float, default=0.75,
                        help='Alpha parameter for GHM loss (default: 0.75). Only used if --loss-type is ghm.')
    
    # Shared loss parameter
    parser.add_argument('--margin', type=float, default=1.0,
                        help='Margin for Ranking Loss (standard or GHM) (default: 1.0).')
    
    # SVRG specific parameters
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate for SVRG optimizer (default: 0.001).')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay for SVRG optimizer (default: 1e-4).')
    
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training and validation (default: 32).')
    
    # Device selection
    parser.add_argument('--device', type=str, default=None, 
                        choices=['cpu', 'cuda', 'mps', 'auto'],
                        help='Device to use: cpu, cuda, mps, or auto (default: auto)')
    
    return parser.parse_args()

def get_device(device_arg):
    """
    確定要使用的設備，支援CUDA、MPS（Apple Silicon）和CPU
    """
    if device_arg == 'cpu':
        return torch.device('cpu')
    
    if device_arg == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    
    if device_arg == 'mps' and torch.backends.mps.is_available():
        return torch.device('mps')
    
    # Auto device selection
    if device_arg is None or device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    # Fallback to CPU if requested device is not available
    print(f"Warning: Requested device '{device_arg}' is not available, using CPU instead.")
    return torch.device('cpu')

def train_with_svrg(args):
    """Trains the model with SVRG optimizer based on the provided arguments."""
    set_seed(args.seed)
    print(f"Using random seed: {args.seed}")
    
    print(f"Starting training - Frequency: {args.frequency}, Material: {args.material}")
    print(f"Loss type: {args.loss_type}")
    if args.loss_type == 'ghm':
        print(f"GHM Params: bins={args.ghm_bins}, alpha={args.ghm_alpha}, margin={args.margin}")
    else:
        print(f"Standard Loss Margin: {args.margin}")
    
    # Setup device with MPS support for Apple Silicon
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    if device.type == 'mps':
        print("Using Apple Metal Performance Shaders (MPS) backend")
    
    # Load data
    print(f"Loading Dataset for {args.frequency} with material {args.material}")
    try:
        dataset = SpectrogramDatasetWithMaterial(
            config.DATA_ROOT,
            config.CLASSES,
            config.SEQ_NUMS,
            args.frequency,
            args.material
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    if len(dataset) == 0:
        print("Dataset is empty. Cannot train model.")
        return None
    
    # Split dataset
    train_size = int(0.70 * len(dataset))
    val_size = len(dataset) - train_size
    if train_size < 4 or val_size < 4:
        print(f"Dataset too small (Total: {len(dataset)}) for train/validation split.")
        return None
    
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    # Create ranking datasets
    train_ranking_dataset = RankingPairDataset(train_dataset)
    val_ranking_dataset = RankingPairDataset(val_dataset)
    
    # Create DataLoaders
    # Note: For MPS compatibility, we might need to adjust num_workers based on the device
    num_workers = 0 if device.type == 'mps' else 4
    
    train_dataloader = torch.utils.data.DataLoader(
        train_ranking_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_ranking_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    if len(train_dataloader) == 0 or len(val_dataloader) == 0:
        print(f"DataLoaders are empty. Cannot train model.")
        return None
    
    # Initialize model
    model = SimpleCNNAudioRanker(n_freqs=dataset.data.shape[2])
    model.to(device)
    
    # Select and initialize loss function
    if args.loss_type == 'ghm':
        criterion = GHMRankingLoss(
            margin=args.margin,
            bins=args.ghm_bins,
            alpha=args.ghm_alpha
        ).to(device)
        # Keep original loss for comparison logging
        original_criterion_for_log = MarginRankingLoss(margin=args.margin).to(device)
    else: # 'standard'
        criterion = MarginRankingLoss(margin=args.margin).to(device)
        original_criterion_for_log = criterion # Log the same loss
    
    # Initialize SVRG optimizers
    optimizer = SVRG_k(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    snapshot = SVRG_Snapshot(model.parameters())
    
    # Training history dictionary
    training_history = {
        'epoch': [],
        'train_loss_main': [],
        'train_loss_standard_log': [],
        'train_accuracy': [],
        'val_loss_main': [],
        'val_loss_standard_log': [],
        'val_accuracy': [],
        'args': vars(args)
    }
    
    # Create save directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"svrg_{args.material}_{args.frequency}_{args.loss_type}_{timestamp}"
    checkpoint_dir = os.path.join(config.SAVE_DIR, 'model_checkpoints', run_name)
    plots_dir = os.path.join(config.SAVE_DIR, 'plots', run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"\n===== Epoch {epoch+1}/{args.epochs} =====")
        
        # Calculate full gradient for snapshot
        model.train()
        snapshot.set_param_groups(optimizer.get_param_groups())
        model.zero_grad()
        
        # Compute the gradient over all samples (snapshot)
        full_grad_loss = 0
        for data1, data2, targets, label1, label2 in train_dataloader:
            data1, data2, targets = data1.to(device), data2.to(device), targets.to(device)
            
            outputs1 = model(data1).view(-1)
            outputs2 = model(data2).view(-1)
            targets = targets.view(-1)
            
            loss = criterion(outputs1, outputs2, targets)
            loss.backward()
            full_grad_loss += loss.item()
        
        full_grad_loss /= len(train_dataloader)
        print(f"Full gradient loss: {full_grad_loss:.4f}")
        
        # Set snapshot gradient to optimizer
        optimizer.set_u(snapshot.get_param_groups())
        
        # Train with SVRG steps
        model.train()
        train_loss_main_accum = 0.0
        train_loss_standard_log_accum = 0.0
        train_correct = 0
        train_total = 0
        
        for i, (data1, data2, targets, label1, label2) in enumerate(train_dataloader):
            data1, data2, targets = data1.to(device), data2.to(device), targets.to(device)
            
            # Reset snapshot model parameters to be the same as current model
            snapshot.set_param_groups(optimizer.get_param_groups())
            
            # Forward pass for main model
            outputs1 = model(data1).view(-1)
            outputs2 = model(data2).view(-1)
            targets = targets.view(-1)
            
            # Calculate main loss (used for backprop)
            loss_main = criterion(outputs1, outputs2, targets)
            
            # Calculate standard loss for logging comparison
            loss_standard_log = original_criterion_for_log(outputs1, outputs2, targets)
            
            # Backward pass for model
            model.zero_grad()
            loss_main.backward()
            
            # Compute gradient on the same data for snapshot model
            snapshot_outputs1 = model(data1).view(-1)
            snapshot_outputs2 = model(data2).view(-1)
            
            snapshot_loss = criterion(snapshot_outputs1, snapshot_outputs2, targets)
            
            # Backward pass for snapshot
            snapshot.zero_grad()
            snapshot_loss.backward()
            
            # Update weights using SVRG
            optimizer.step(snapshot.get_param_groups())
            
            train_loss_main_accum += loss_main.item()
            train_loss_standard_log_accum += loss_standard_log.item()
            
            predictions = (outputs1 > outputs2) == (targets > 0)
            train_correct += predictions.sum().item()
            train_total += targets.size(0)
            
            if (i+1) % 10 == 0: # Print progress
                log_str = f"Batch {i+1}/{len(train_dataloader)} | Main Loss: {loss_main.item():.4f}"
                if args.loss_type == 'ghm':
                    log_str += f" | Std Loss (log): {loss_standard_log.item():.4f}"
                print(log_str)
        
        # Validation phase
        model.eval()
        val_loss_main_accum = 0.0
        val_loss_standard_log_accum = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data1, data2, targets, label1, label2 in val_dataloader:
                data1, data2, targets = data1.to(device), data2.to(device), targets.to(device)
                
                outputs1 = model(data1).view(-1)
                outputs2 = model(data2).view(-1)
                targets = targets.view(-1)
                
                loss_main = criterion(outputs1, outputs2, targets)
                loss_standard_log = original_criterion_for_log(outputs1, outputs2, targets)
                
                val_loss_main_accum += loss_main.item()
                val_loss_standard_log_accum += loss_standard_log.item()
                
                predictions = (outputs1 > outputs2) == (targets > 0)
                val_correct += predictions.sum().item()
                val_total += targets.size(0)
        
        # Calculate average losses and accuracy
        avg_train_loss_main = train_loss_main_accum / len(train_dataloader) if len(train_dataloader) > 0 else float('inf')
        avg_train_loss_standard = train_loss_standard_log_accum / len(train_dataloader) if len(train_dataloader) > 0 else float('inf')
        train_accuracy = 100.0 * train_correct / train_total if train_total > 0 else 0.0
        
        avg_val_loss_main = val_loss_main_accum / len(val_dataloader) if len(val_dataloader) > 0 else float('inf')
        avg_val_loss_standard = val_loss_standard_log_accum / len(val_dataloader) if len(val_dataloader) > 0 else float('inf')
        val_accuracy = 100.0 * val_correct / val_total if val_total > 0 else 0.0
        
        print(f"Epoch {epoch+1} Summary")
        print(f"  Training   - Main Loss: {avg_train_loss_main:.4f}, Std Loss (log): {avg_train_loss_standard:.4f}, Accuracy: {train_accuracy:.2f}%")
        print(f"  Validation - Main Loss: {avg_val_loss_main:.4f}, Std Loss (log): {avg_val_loss_standard:.4f}, Accuracy: {val_accuracy:.2f}%")
        
        # Record history
        training_history['epoch'].append(epoch + 1)
        training_history['train_loss_main'].append(avg_train_loss_main)
        training_history['train_loss_standard_log'].append(avg_train_loss_standard)
        training_history['train_accuracy'].append(train_accuracy)
        training_history['val_loss_main'].append(avg_val_loss_main)
        training_history['val_loss_standard_log'].append(avg_val_loss_standard)
        training_history['val_accuracy'].append(val_accuracy)
        
        # Save checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0 or epoch == args.epochs - 1:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"model_epoch_{epoch+1}.pt"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'train_loss_main': avg_train_loss_main,
                'val_loss_main': avg_val_loss_main,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'args': vars(args)
            }, checkpoint_path)
            print(f"Model checkpoint saved to: {checkpoint_path}")
    
    # Plot training history
    plot_path = os.path.join(plots_dir, f'training_history_{timestamp}.png')
    plot_training_history(training_history, plot_path)
    print(f"Training history plot saved to: {plot_path}")
    
    # Save training history dictionary
    history_path = os.path.join(checkpoint_dir, f"training_history_{timestamp}.pkl")
    with open(history_path, 'wb') as f:
        pickle.dump(training_history, f)
    print(f"Training history saved to: {history_path}")
    
    return model, training_history

def main():
    """Main function to parse arguments and initiate training."""
    args = parse_arguments()
    
    # 系統資訊輸出
    print(f"System: {platform.system()} {platform.version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    available_frequencies = ['500hz', '1000hz', '3000hz']
    
    if args.frequency == 'all':
        print("\nTraining models for all available frequencies with SVRG:")
        results = {}
        for freq in available_frequencies:
            print(f"\n--- Training Frequency: {freq} ---")
            # Create a copy of args to modify frequency for each run
            current_args = argparse.Namespace(**vars(args))
            current_args.frequency = freq
            try:
                model, history = train_with_svrg(current_args)
                results[freq] = {'success': True, 'history': history}
            except Exception as e:
                print(f"ERROR during training for frequency {freq}: {e}")
                results[freq] = {'success': False, 'error': str(e)}
        print("\n--- Finished training all frequencies --- ")
        # Print summary of results
        for freq, result in results.items():
            status = "Success" if result['success'] else f"Failed ({result.get('error', '')})"
            print(f"Frequency {freq}: {status}")
    
    else:
        print(f"\n--- Training Frequency: {args.frequency} with SVRG ---")
        try:
            train_with_svrg(args)
            print(f"\n--- Finished training {args.frequency} ---")
        except Exception as e:
            print(f"ERROR during training for frequency {args.frequency}: {e}")

if __name__ == "__main__":
    main() 