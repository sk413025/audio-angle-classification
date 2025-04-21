#!/usr/bin/env python
"""
Analyze per-checkpoint TracIn influence scores.

This script is a backward compatibility layer that calls the functionality from the tracin module.
"""

import os
import sys
import argparse

# Ensure the tracin module can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from the new module
from tracin.utils.influence_utils import load_influence_scores


def main():
    """Main function."""
    # Display compatibility notice
    print("Note: This script is maintained for backward compatibility. Consider using the tracin module directly:")
    print("  python -m tracin.utils.influence_tools [arguments]")
    print("Starting TracIn module...\n")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Analyze per-checkpoint TracIn influence scores")
    parser.add_argument('--metadata-file', type=str, required=True,
                        help='Path to per-checkpoint influence metadata file')
    args = parser.parse_args()
    
    # Load per-checkpoint influence scores
    try:
        print(f"Loading per-checkpoint influence scores from {args.metadata_file}")
        influence_scores = load_influence_scores(args.metadata_file)
        print(f"Loaded {len(influence_scores)} influence scores")
        
        # Count the number of checkpoints
        checkpoint_counts = {}
        for sample_id, scores in influence_scores.items():
            for key in scores.keys():
                if "model_epoch" in key:
                    parts = key.split("model_epoch_")
                    if len(parts) > 1:
                        checkpoint = parts[1].split(".pt")[0]
                        checkpoint_counts[checkpoint] = checkpoint_counts.get(checkpoint, 0) + 1
        
        # Display checkpoint counts
        print("\nCheckpoint statistics:")
        for checkpoint, count in sorted(checkpoint_counts.items(), key=lambda x: int(x[0])):
            print(f"  Epoch {checkpoint}: {count} influence scores")
            
    except Exception as e:
        print(f"Error analyzing per-checkpoint influence scores: {e}")


if __name__ == "__main__":
    main() 