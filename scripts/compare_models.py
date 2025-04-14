#!/usr/bin/env python
"""
Model Comparison Script

This script compares multiple trained models on various metrics.

Usage:
    python scripts/compare_models.py --model-paths <model1.pt> <model2.pt> --frequency <freq>
    
Example:
    python scripts/compare_models.py \
        --model-paths saved_models/ce_1000hz_model.pt saved_models/ghm_1000hz_model.pt \
        --frequency 1000hz
"""

import os
import argparse
import sys

# Add the root directory to the Python path to ensure imports work correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.analysis.comparison import compare_models

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compare multiple trained models")
    
    parser.add_argument('--model-paths', type=str, nargs='+', required=True,
                        help='Paths to the model files to compare')
    
    parser.add_argument('--model-names', type=str, nargs='+', default=None,
                        help='Optional names for the models (default: extracted from filenames)')
    
    parser.add_argument('--frequency', type=str, required=True,
                        choices=['500hz', '1000hz', '3000hz'],
                        help='Frequency data to use for comparison')
    
    parser.add_argument('--material', type=str, default='metal',
                        help='Material type (default: metal)')
    
    parser.add_argument('--metrics', type=str, nargs='+', 
                        default=['accuracy', 'confusion', 'angle_error'],
                        choices=['accuracy', 'confusion', 'angle_error', 'parameters', 'training_time'],
                        help='Metrics to compare (default: accuracy, confusion, angle_error)')
    
    parser.add_argument('--output-dir', type=str, default='results/model_comparison',
                        help='Directory to save comparison results')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def main():
    """Main function for model comparison."""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # If model names not provided, extract from filenames
    if not args.model_names or len(args.model_names) != len(args.model_paths):
        args.model_names = [os.path.splitext(os.path.basename(path))[0] for path in args.model_paths]
    
    print(f"Comparing {len(args.model_paths)} models:")
    for name, path in zip(args.model_names, args.model_paths):
        print(f"  - {name}: {path}")
    
    print(f"Frequency: {args.frequency}, Material: {args.material}")
    print(f"Metrics: {', '.join(args.metrics)}")
    
    # Perform the comparison
    results = compare_models(
        model_paths=args.model_paths,
        model_names=args.model_names,
        frequency=args.frequency,
        material=args.material,
        metrics=args.metrics,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    print(f"Comparison complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()