#!/usr/bin/env python
"""
Generate a comprehensive TracIn analysis report.

This script is a backward compatibility layer that calls the functionality from the tracin module.
"""

import os
import sys
import argparse

# Ensure the tracin module can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from the new module
from tracin.utils.influence_utils import get_harmful_samples
from tracin.utils.visualization import plot_harmful_samples


def main():
    """Main function."""
    # Display compatibility notice
    print("Note: This script is maintained for backward compatibility. Consider using the tracin module directly:")
    print("  python -m tracin.utils.report_generator [arguments]")
    print("Starting TracIn module...\n")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate a comprehensive TracIn analysis report")
    parser.add_argument('--metadata-file', type=str, required=True,
                        help='Path to influence metadata file')
    parser.add_argument('--output-dir', type=str, default='tracin_reports',
                        help='Directory to save report files')
    parser.add_argument('--threshold', type=float, default=-5.0,
                        help='Threshold for harmful influence scores')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate report
    try:
        print(f"Analyzing influence data from {args.metadata_file}")
        
        # Find harmful samples
        harmful_samples = get_harmful_samples(
            metadata_file=args.metadata_file,
            threshold=args.threshold,
            min_occurrences=1
        )
        
        print(f"Found {len(harmful_samples)} potentially harmful samples")
        
        # Create visualizations
        if harmful_samples:
            plot_path = os.path.join(args.output_dir, "harmful_samples.png")
            plot_harmful_samples(
                harmful_samples=harmful_samples,
                save_path=plot_path,
                title="Harmful Training Samples",
                max_samples=min(20, len(harmful_samples))
            )
            print(f"Saved harmful samples visualization to {plot_path}")
            
            # Generate a text report
            report_path = os.path.join(args.output_dir, "tracin_report.txt")
            with open(report_path, 'w') as f:
                f.write("TracIn Analysis Report\n")
                f.write("=====================\n\n")
                f.write(f"Metadata file: {args.metadata_file}\n")
                f.write(f"Threshold: {args.threshold}\n\n")
                f.write(f"Found {len(harmful_samples)} potentially harmful samples\n\n")
                
                f.write("Top 10 Harmful Samples:\n")
                f.write("----------------------\n\n")
                
                for i, sample in enumerate(harmful_samples[:10]):
                    f.write(f"{i+1}. {sample['sample_id']}\n")
                    f.write(f"   Negative occurrences: {sample['negative_occurrences']}\n")
                    f.write(f"   Average influence: {sample['average_influence']:.4f}\n")
                    
                    f.write("   Influential on test samples:\n")
                    for test_id, score in sample['examples']:
                        f.write(f"   - {test_id}: {score:.4f}\n")
                    f.write("\n")
            
            print(f"Generated report saved to {report_path}")
                
    except Exception as e:
        print(f"Error generating TracIn report: {e}")


if __name__ == "__main__":
    main() 