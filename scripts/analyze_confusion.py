#!/usr/bin/env python
"""
Confusion Matrix Analysis Script

This script analyzes model performance using confusion matrices and angle accuracy metrics.

Usage:
    python scripts/analyze_confusion.py --model-path <model_path> --frequency <freq> --material <material>
    
Example:
    python scripts/analyze_confusion.py --model-path saved_models/ghm_1000hz_model.pt --frequency 1000hz
"""

import os
import argparse
import sys

# Add the root directory to the Python path to ensure imports work correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.analysis.model import analyze_confusion_matrix

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze model performance using confusion matrices")
    
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the model file')
    
    parser.add_argument('--frequency', type=str, required=True,
                        choices=['500hz', '1000hz', '3000hz'],
                        help='Frequency data to use for analysis')
    
    parser.add_argument('--material', type=str, default='metal',
                        help='Material type (default: metal)')
    
    parser.add_argument('--output-dir', type=str, default='results/confusion_matrix',
                        help='Directory to save analysis results')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def main():
    """Main function for confusion matrix analysis."""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Analyzing model: {args.model_path}")
    print(f"Frequency: {args.frequency}, Material: {args.material}")
    
    # Perform the analysis
    results = analyze_confusion_matrix(
        model_path=args.model_path,
        frequency=args.frequency,
        material=args.material,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    if results:
        print(f"Analysis complete. Results saved to {args.output_dir}")
    else:
        print("Analysis failed. Check the error messages above.")

if __name__ == "__main__":
    main()