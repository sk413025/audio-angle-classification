#!/usr/bin/env python
"""
Visualize TracIn influence scores.

This script is a backward compatibility layer that calls the functionality from the tracin module.
"""

import os
import sys
import argparse

# Ensure the tracin module can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from the new module
from tracin.utils.visualization import plot_influence_distribution, plot_sample_influence_heatmap


def main():
    """Main function."""
    # Display compatibility notice
    print("Note: This script is maintained for backward compatibility. Consider using the tracin module directly:")
    print("  python -m tracin.utils.visualization_tools [arguments]")
    print("Starting TracIn module...\n")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Visualize TracIn influence scores")
    parser.add_argument('--metadata-file', type=str, required=True,
                        help='Path to influence metadata file')
    parser.add_argument('--output-dir', type=str, default='tracin_visualizations',
                        help='Directory to save visualizations')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate visualizations
    plot_influence_distribution(
        metadata_file=args.metadata_file,
        save_path=os.path.join(args.output_dir, 'influence_distribution.png')
    )
    
    plot_sample_influence_heatmap(
        metadata_file=args.metadata_file,
        save_path=os.path.join(args.output_dir, 'influence_heatmap.png')
    )
    
    print(f"Visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main() 