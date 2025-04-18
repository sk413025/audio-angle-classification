"""
Main training script for the angle classification model.

Supports training with standard MarginRankingLoss or GHM Loss.
Allows configuration via command-line arguments.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from datetime import datetime
import numpy as np
import pickle
import json

import config
from datasets import (
    AudioSpectrumDataset, 
    RankingPairDataset, 
    GHMAwareRankingDataset,
    SpectrogramDatasetWithMaterial, 
    DatasetConfig
)
# Assuming the model is in models/resnet_ranker.py now or will be moved there
# We'll use the existing import for now and adjust later if needed.
from models.resnet_ranker import SimpleCNNAudioRanker as ResNetAudioRanker

# Import loss functions
from torch.nn import MarginRankingLoss
from losses.ghm_loss import GHMRankingLoss

# Import utils
from utils.common_utils import worker_init_fn, set_seed
from utils import ghm_utils
from utils.debugging_utils import verify_batch_consistency # Import the function

# Import visualization modules
from utils.visualization import plot_training_history, plot_ghm_statistics

def parse_arguments():
    """Parses command-line arguments for the training script.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Train Angle Classification Model')

    parser.add_argument('--frequency', type=str, required=True, 
                        choices=['500hz', '1000hz', '3000hz', 'all'],
                        help='Frequency data to use for training (or \'all\').')
    
    parser.add_argument('--material', type=str, default=config.MATERIAL, 
                        help=f'Material type (default: {config.MATERIAL} from config.py).')

    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs (default: 30).')

    parser.add_argument('--checkpoint-interval', type=int, default=5,
                        help='Save model checkpoint every N epochs (default: 5).')

    parser.add_argument('--seed', type=int, default=config.SEED if hasattr(config, 'SEED') else 42,
                        help='Random seed for reproducibility (default: from config.py or 42).')

    parser.add_argument('--loss-type', type=str, default='ghm',
                        choices=['standard', 'ghm'],
                        help='Type of loss function to use (default: ghm).')

    # GHM specific arguments
    parser.add_argument('--ghm-bins', type=int, default=10,
                        help='Number of bins for GHM loss (default: 10). Only used if --loss-type is ghm.')
    parser.add_argument('--ghm-alpha', type=float, default=0.75,
                        help='Alpha parameter for GHM loss (default: 0.75). Only used if --loss-type is ghm.')

    # Shared loss parameter
    parser.add_argument('--margin', type=float, default=config.MARGIN if hasattr(config, 'MARGIN') else 1.0,
                        help='Margin for Ranking Loss (standard or GHM) (default: from config.py or 1.0).')

    # Optional overrides for config values
    parser.add_argument('--learning-rate', type=float, default=config.LEARNING_RATE if hasattr(config, 'LEARNING_RATE') else 0.001,
                        help='Learning rate for Adam optimizer (default: from config.py or 0.001).')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE if hasattr(config, 'BATCH_SIZE') else 32,
                        help='Batch size for training and validation (default: from config.py or 32).')
    parser.add_argument('--weight-decay', type=float, default=config.WEIGHT_DECAY if hasattr(config, 'WEIGHT_DECAY') else 1e-4,
                        help='Weight decay for Adam optimizer (default: from config.py or 1e-4).')

    # Boolean flags
    parser.add_argument('--verify-consistency', action='store_true',
                        help='Verify batch consistency by saving initial batches.')
                        
    # 添加數據集配置參數
    parser.add_argument('--dataset-version', type=str, default='1.0',
                        help='Dataset version to use.')
    parser.add_argument('--exclusions-file', type=str, default=None,
                        help='Path to sample exclusion list file.')
    parser.add_argument('--metadata-file', type=str, default=None,
                        help='Path to sample metadata file.')
    parser.add_argument('--track-ghm-samples', action='store_true',
                        help='Track sample assignments to GHM bins.')

    args = parser.parse_args()
    return args

def train_model(args):
    """Trains the model based on the provided arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        tuple: (trained_model, training_history) or None if training fails.
    """
    set_seed(args.seed)
    print(f"Using random seed: {args.seed}")

    print(f"Starting training - Frequency: {args.frequency}, Material: {args.material}")
    print(f"Loss type: {args.loss_type}")
    if args.loss_type == 'ghm':
        print(f"GHM Params: bins={args.ghm_bins}, alpha={args.ghm_alpha}, margin={args.margin}")
    else:
        print(f"Standard Loss Margin: {args.margin}")

    # Setup device
    device = config.DEVICE
    print(f"Using device: {device}")
    
    # 加載數據集設置
    dataset_config = DatasetConfig(
        version=args.dataset_version,
        exclusions=args.exclusions_file,
        metadata_file=args.metadata_file,
        description=f"Training run for {args.material}_{args.frequency}_{args.loss_type}"
    )
    
    # 確保元數據目錄存在
    metadata_dir = os.path.dirname(dataset_config.get_metadata_path())
    os.makedirs(metadata_dir, exist_ok=True)
    exclusions_dir = os.path.dirname(dataset_config.get_exclusion_path())
    os.makedirs(exclusions_dir, exist_ok=True)

    # Load data
    print(f"Loading Dataset for {args.frequency} with material {args.material}")
    try:
        # 使用新的數據集類
        dataset = AudioSpectrumDataset(
            data_root=config.DATA_ROOT,
            classes=config.CLASSES,
            selected_seqs=config.SEQ_NUMS,
            selected_freq=args.frequency,
            material=args.material,
            exclusion_file=dataset_config.get_exclusion_path(),
            metadata_file=dataset_config.get_metadata_path()
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
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    # Create ranking datasets
    # 如果是使用GHM，創建支持樣本跟蹤的排序數據集
    if args.loss_type == 'ghm' and args.track_ghm_samples:
        train_ranking_dataset = GHMAwareRankingDataset(train_dataset)
        val_ranking_dataset = GHMAwareRankingDataset(val_dataset)
        print("Using GHM-aware ranking dataset with sample tracking")
    else:
        train_ranking_dataset = RankingPairDataset(train_dataset)
        val_ranking_dataset = RankingPairDataset(val_dataset)

    # Create DataLoaders
    min_ranking_size = min(len(train_ranking_dataset), len(val_ranking_dataset))
    # Use batch size from args, ensuring it's not larger than the smallest dataset
    effective_batch_size = min(args.batch_size, min_ranking_size if min_ranking_size > 0 else args.batch_size)
    
    if effective_batch_size <= 0:
         print(f"Effective batch size is {effective_batch_size}. Cannot create DataLoader.")
         return None

    print(f"Train ranking pairs: {len(train_ranking_dataset)}")
    print(f"Validation ranking pairs: {len(val_ranking_dataset)}")
    print(f"Using batch size: {effective_batch_size}")

    train_dataloader = DataLoader(
        train_ranking_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=4, # Consider making num_workers configurable
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(args.seed)
    )

    val_dataloader = DataLoader(
        val_ranking_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=4, # Consider making num_workers configurable
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(args.seed)
    )

    if len(train_dataloader) == 0 or len(val_dataloader) == 0:
        print(f"DataLoaders are empty. Cannot train model.")
        return None

    # Verify batch consistency if requested
    if args.verify_consistency:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        consistency_dir = os.path.join(config.SAVE_DIR, 'batch_consistency', f"{args.material}_{args.frequency}_{args.loss_type}")
        os.makedirs(consistency_dir, exist_ok=True)
        train_batch_file = os.path.join(consistency_dir, f"train_batches_{timestamp}_seed{args.seed}.pkl")
        verify_batch_consistency(train_dataloader, num_batches=3, save_path=train_batch_file)

    # Initialize model
    model = ResNetAudioRanker(n_freqs=dataset.data.shape[2])
    model.to(device)

    # Select and initialize loss function
    if args.loss_type == 'ghm':
        criterion = GHMRankingLoss(
            margin=args.margin,
            bins=args.ghm_bins,
            alpha=args.ghm_alpha
        ).to(device)
        # Keep original loss for comparison logging if needed
        original_criterion_for_log = MarginRankingLoss(margin=args.margin).to(device)
    else: # 'standard'
        criterion = MarginRankingLoss(margin=args.margin).to(device)
        original_criterion_for_log = criterion # Log the same loss

    # Optimizer (use args for LR and weight decay)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )

    # Scheduler (monitor validation loss - use the main criterion's loss)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # Training history dictionary
    training_history = {
        'epoch': [],
        'train_loss_main': [], # Loss from the main criterion used for backprop
        'train_loss_standard_log': [], # Standard loss logged for comparison
        'train_accuracy': [],
        'val_loss_main': [],
        'val_loss_standard_log': [],
        'val_accuracy': [],
        'args': vars(args) # Store arguments used for this run
    }

    # Create save directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.loss_type == 'ghm':
        run_name = f"{args.material}_{args.frequency}_{args.loss_type}_alpha{args.ghm_alpha}_{timestamp}"
    else:
        run_name = f"{args.material}_{args.frequency}_{args.loss_type}_{timestamp}"
    checkpoint_dir = os.path.join(config.SAVE_DIR, 'model_checkpoints', run_name)
    plots_dir = os.path.join(config.SAVE_DIR, 'plots', run_name)
    stats_dir = os.path.join(config.SAVE_DIR, 'stats', run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    if args.loss_type == 'ghm':
         os.makedirs(stats_dir, exist_ok=True) # Only create stats dir if using GHM

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss_main_accum = 0.0
        train_loss_standard_log_accum = 0.0
        train_correct = 0
        train_total = 0

        print(f"\n===== Epoch {epoch+1}/{args.epochs} =====")

        for i, batch in enumerate(train_dataloader):
            # 處理資料批次，支援新的數據集返回格式
            if len(batch) == 7:  # 新格式：(data1, data2, targets, label1, label2, sample_id1, sample_id2)
                data1, data2, targets, label1, label2, sample_id1, sample_id2 = batch
            else:  # 舊格式：(data1, data2, targets, label1, label2)
                data1, data2, targets, label1, label2 = batch
                sample_id1 = sample_id2 = None
                
            data1, data2, targets = data1.to(device), data2.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            outputs1 = model(data1).view(-1)
            outputs2 = model(data2).view(-1)
            targets = targets.view(-1)

            # Calculate main loss (used for backprop)
            if args.loss_type == 'ghm' and args.track_ghm_samples and sample_id1 is not None:
                # 如果啟用了樣本跟蹤，傳遞樣本ID
                loss_main = criterion(outputs1, outputs2, targets, sample_id1, sample_id2)
            else:
                loss_main = criterion(outputs1, outputs2, targets)
            
            # Calculate standard loss for logging comparison
            loss_standard_log = original_criterion_for_log(outputs1, outputs2, targets)

            loss_main.backward()
            optimizer.step()

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

            # GHM statistics (conditional)
            if args.loss_type == 'ghm' and (i == 0 or i == len(train_dataloader) - 1):
                # Use the visualization module to plot GHM statistics
                plot_ghm_statistics(
                    criterion, outputs1, outputs2, targets,
                    save_dir=stats_dir, 
                    name=f'epoch{epoch+1}_batch{i}_{timestamp}'
                )
                
                # 如果啟用了樣本跟蹤，保存bin分配
                if args.track_ghm_samples and hasattr(criterion, 'get_bin_statistics'):
                    bin_stats = criterion.get_bin_statistics()
                    bin_stats_path = os.path.join(stats_dir, f'bin_stats_epoch{epoch+1}_batch{i}.json')
                    with open(bin_stats_path, 'w') as f:
                        json.dump(bin_stats, f, indent=2)
                    print(f"Saved bin statistics to {bin_stats_path}")
                    
                    # 如果使用GHMAwareRankingDataset，保存bin分配
                    if isinstance(train_ranking_dataset, GHMAwareRankingDataset):
                        # 迭代每個bin，記錄分配給該bin的樣本
                        for bin_idx in range(args.ghm_bins):
                            # 獲取該bin中的樣本對
                            bin_samples = criterion.get_bin_samples(bin_idx)
                            for sample_id1, sample_id2 in bin_samples:
                                # 將bin分配記錄到樣本元數據中
                                dataset.record_ghm_bin(sample_id1, epoch + 1, bin_idx)
                                dataset.record_ghm_bin(sample_id2, epoch + 1, bin_idx)
                        
                        # 保存更新的元數據
                        dataset.save_metadata(dataset_config.get_metadata_path())
                        print(f"Updated sample metadata with GHM bin assignments")

        # Validation phase
        model.eval()
        val_loss_main_accum = 0.0
        val_loss_standard_log_accum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_dataloader:
                # 處理資料批次，支援新的數據集返回格式
                if len(batch) == 7:  # 新格式
                    data1, data2, targets, label1, label2, _, _ = batch
                else:  # 舊格式
                    data1, data2, targets, label1, label2 = batch
                
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

        # Step the scheduler based on the main validation loss
        scheduler.step(avg_val_loss_main)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Current learning rate: {current_lr:.6f}")

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
                f"model_epoch_{epoch+1}.pt" # Simplified name, details in the file
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss_main': avg_train_loss_main,
                'val_loss_main': avg_val_loss_main,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'args': vars(args) # Save args used for this checkpoint
            }, checkpoint_path)
            print(f"Model checkpoint saved to: {checkpoint_path}")

    # --- End of Training Loop ---

    # Plot training history using the visualization module
    plot_path = os.path.join(plots_dir, f'training_history_{timestamp}.png')
    plot_training_history(training_history, plot_path)
    print(f"Training history plot saved to: {plot_path}")

    # Save training history dictionary
    history_path = os.path.join(checkpoint_dir, f"training_history_{timestamp}.pkl") # Changed to pkl
    with open(history_path, 'wb') as f:
        pickle.dump(training_history, f)
    print(f"Training history saved to: {history_path}")

    # 保存最終的數據集元數據
    dataset.save_metadata(dataset_config.get_metadata_path())
    print(f"Final dataset metadata saved to: {dataset_config.get_metadata_path()}")

    # 保存數據集配置
    config_path = os.path.join(stats_dir, f"dataset_config_{timestamp}.json")
    dataset_config.save(config_path)
    print(f"Dataset configuration saved to: {config_path}")

    return model, training_history

def main():
    """Main function to parse arguments and initiate training."""
    args = parse_arguments()

    available_frequencies = ['500hz', '1000hz', '3000hz']

    if args.frequency == 'all':
        print("\nTraining models for all available frequencies:")
        results = {}
        for freq in available_frequencies:
            print(f"\n--- Training Frequency: {freq} ---")
            # Create a copy of args to modify frequency for each run
            current_args = argparse.Namespace(**vars(args))
            current_args.frequency = freq
            try:
                 model, history = train_model(current_args)
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
        print(f"\n--- Training Frequency: {args.frequency} ---")
        try:
            train_model(args)
            print(f"\n--- Finished training {args.frequency} ---")
        except Exception as e:
            print(f"ERROR during training for frequency {args.frequency}: {e}")

if __name__ == "__main__":
    main()
