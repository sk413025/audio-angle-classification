#!/usr/bin/env python
"""
GHM Analysis Script

This script analyzes GHM training results, providing visualizations and metrics
for understanding model training behavior with the GHM loss function.

Usage:
    python scripts/analyze_ghm.py --base-dir <base_dir> --output-dir <output_dir> --frequency <freq>
    
Example:
    python scripts/analyze_ghm.py --base-dir saved_models --output-dir results/ghm --frequency 1000hz
"""

import os
import argparse
import sys

# Add the root directory to the Python path to ensure imports work correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.analysis.ghm import (
    analyze_ghm_results,
    analyze_ghm_details
)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze GHM training results")
    
    parser.add_argument('--base-dir', type=str, default='saved_models',
                        help='Base directory containing training results')
    
    parser.add_argument('--output-dir', type=str, default='results/ghm',
                        help='Directory to save analysis results')
    
    parser.add_argument('--frequency', type=str, default='1000hz',
                        choices=['500hz', '1000hz', '3000hz', 'all'],
                        help='Frequency to analyze (or "all")')
    
    parser.add_argument('--detailed', action='store_true',
                        help='Perform detailed analysis')
    
    parser.add_argument('--param-type', type=str, default=None,
                        choices=['alpha', 'bins', None],
                        help='Parameter type for comparison in detailed analysis')
    
    parser.add_argument('--param-value', type=str, default=None,
                        help='Parameter value for detailed analysis')
    
    return parser.parse_args()

def main():
    """Main function for GHM analysis."""
    args = parse_arguments()
    
    base_dir = args.base_dir
    output_dir = args.output_dir
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if args.frequency == 'all':
        frequencies = ['500hz', '1000hz', '3000hz']
    else:
        frequencies = [args.frequency]
    
    # Perform the analysis for each frequency
    for frequency in frequencies:
        freq_output_dir = os.path.join(output_dir, frequency)
        os.makedirs(freq_output_dir, exist_ok=True)
        
        print(f"Analyzing frequency: {frequency}")
        
        # Basic GHM results analysis
        print(f"Performing GHM results analysis...")
        results = analyze_ghm_results(base_dir, freq_output_dir)
        
        # Detailed analysis if requested
        if args.detailed:
            print(f"Performing detailed GHM analysis...")
            detailed_output_dir = os.path.join(freq_output_dir, 'detailed')
            analyze_ghm_details(
                base_dir,
                detailed_output_dir,
                frequency=frequency,
                param_type=args.param_type,
                param_value=args.param_value
            )
    
    print(f"Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()