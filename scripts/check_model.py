#!/usr/bin/env python3
"""
Script to check and analyze PyTorch model checkpoint files.
"""

import argparse
import json
import torch
from pathlib import Path


def check_model(model_path: str) -> dict:
    """
    Check the contents of a PyTorch model checkpoint file.
    
    Args:
        model_path: Path to the model checkpoint file
        
    Returns:
        Dictionary containing information about the model checkpoint
    """
    checkpoint = torch.load(model_path, weights_only=False)
    
    info = {
        "has_model_state_dict": "model_state_dict" in checkpoint,
        "has_optimizer_state_dict": "optimizer_state_dict" in checkpoint,
        "has_training_history": "training_history" in checkpoint,
    }
    
    if info["has_model_state_dict"]:
        info["model_state_dict_keys"] = list(checkpoint["model_state_dict"].keys())
        
    if info["has_optimizer_state_dict"]:
        info["optimizer_state_dict_keys"] = list(checkpoint["optimizer_state_dict"].keys())
        
    if info["has_training_history"]:
        info["training_history_epochs"] = len(checkpoint["training_history"])
        
    return info


def main():
    parser = argparse.ArgumentParser(description="Check PyTorch model checkpoint files")
    parser.add_argument("--model-path", required=True, help="Path to the model checkpoint file")
    parser.add_argument("--output-file", required=True, help="Path to save the output")
    
    args = parser.parse_args()
    
    # Check if model file exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    # Analyze model
    info = check_model(args.model_path)
    
    # Save results
    with open(args.output_file, "w") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()