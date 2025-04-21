#!/usr/bin/env python
"""
Analyze TracIn influence scores.

This script is a backward compatibility layer that calls the functionality from the tracin module.
"""

import os
import sys
import argparse

# Ensure the tracin module can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from the new module
from tracin.utils.influence_utils import load_influence_scores, get_harmful_samples


def main():
    """Main function."""
    # Display compatibility notice
    print("Note: This script is maintained for backward compatibility. Consider using the tracin module directly:")
    print("  python -m tracin.utils.influence_tools [arguments]")
    print("Starting TracIn module...\n")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Analyze TracIn influence scores")
    parser.add_argument('--metadata-file', type=str, required=True,
                        help='Path to influence metadata file')
    parser.add_argument('--threshold', type=float, default=-5.0,
                        help='Threshold for harmful influence scores')
    args = parser.parse_args()
    
    # Load scores
    influence_scores = load_influence_scores(args.metadata_file)
    print(f"Loaded {len(influence_scores)} influence scores")
    
    # Find harmful samples
    harmful_samples = get_harmful_samples(
        metadata_file=args.metadata_file,
        threshold=args.threshold,
        min_occurrences=1
    )
    
    # Print results
    print(f"Found {len(harmful_samples)} potentially harmful samples")
    for i, sample in enumerate(harmful_samples[:10]):
        print(f"\n{i+1}. {sample['sample_id']}")
        print(f"   Negative occurrences: {sample['negative_occurrences']}")
        print(f"   Average influence: {sample['average_influence']:.4f}")


if __name__ == "__main__":
    main() 