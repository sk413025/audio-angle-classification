#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.seed_utils import seed_everything, worker_init_fn
from losses.ghm_loss import GHMC_Loss, GHMR_Loss
from models.resnet_ranker import ResNetRanker
from datasets import AudioDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Train audio angle classification model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='saved_models', help='Output directory')
    parser.add_argument('--loss', type=str, default='ghmc', help='Loss function: ghmc, ghmr, or ce')
    parser.add_argument('--ghm_bins', type=int, default=10, help='Number of bins for GHM loss')
    parser.add_argument('--ghm_alpha', type=float, default=0.75, help='Alpha for GHM loss')
    parser.add_argument('--ghm_momentum', type=float, default=0.9, help='Momentum for GHM loss')
    return parser.parse_args()

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, output_dir):
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
    
    return model

def main():
    args = parse_args()
    
    # Set random seed
    seed_everything(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = ResNetRanker(num_classes=360)
    model = model.to(device)
    
    # Setup dataloaders
    train_dataset = AudioDataset(root_dir=args.data_dir, split='train')
    val_dataset = AudioDataset(root_dir=args.data_dir, split='val')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        worker_init_fn=worker_init_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        worker_init_fn=worker_init_fn
    )
    
    # Setup loss function
    if args.loss == 'ghmc':
        criterion = GHMC_Loss(
            bins=args.ghm_bins,
            alpha=args.ghm_alpha,
            momentum=args.ghm_momentum
        )
    elif args.loss == 'ghmr':
        criterion = GHMR_Loss(
            bins=args.ghm_bins,
            alpha=args.ghm_alpha,
            momentum=args.ghm_momentum
        )
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train model
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        output_dir=args.output_dir
    )
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))
    print(f'Model saved to {args.output_dir}')

if __name__ == '__main__':
    main() 