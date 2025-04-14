#!/usr/bin/env python
"""
Model Analysis Script

This script analyzes trained model structure and performance metrics.

Usage:
    python scripts/analyze_model.py --model-path <model_path> --frequency <freq>
    
Example:
    python scripts/analyze_model.py --model-path saved_models/ghm_1000hz_model.pt --frequency 1000hz
"""

import os
import argparse
import sys

# Add the root directory to the Python path to ensure imports work correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.analysis.model import analyze_model_structure

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze model structure and performance")
    
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the model file')
    
    parser.add_argument('--frequency', type=str, default=None,
                        choices=['500hz', '1000hz', '3000hz', None],
                        help='Frequency to use for input analysis (optional)')
    
    parser.add_argument('--output-dir', type=str, default='results/model_analysis',
                        help='Directory to save analysis results')
    
    parser.add_argument('--detailed', action='store_true',
                        help='Perform detailed analysis with test data')
    
    return parser.parse_args()

def main():
    """Main function for model analysis."""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Analyzing model: {args.model_path}")
    if args.frequency:
        print(f"Using frequency: {args.frequency} for input analysis")
    
    # Perform model structure analysis
    model_info = analyze_model_structure(
        model_path=args.model_path,
        frequency=args.frequency,
        output_dir=args.output_dir,
        detailed=args.detailed
    )
    
    print(f"Analysis complete. Results saved to {args.output_dir}")
    
    # Print summary of model structure
    if model_info:
        print("\nModel Summary:")
        print(f"Model Type: {model_info.get('model_type', 'Unknown')}")
        print(f"Input Shape: {model_info.get('input_shape', 'Unknown')}")
        print(f"Output Shape: {model_info.get('output_shape', 'Unknown')}")
        print(f"Parameter Count: {model_info.get('param_count', 'Unknown'):,}")

if __name__ == "__main__":
    main()