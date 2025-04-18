#!/usr/bin/env python
"""
GHM Sample Analysis Tool

This script analyzes the samples assigned to different GHM bins during training.
It helps identify problematic samples for potential exclusion or investigation.

Usage:
    python scripts/analyze_ghm_samples.py --metadata-file <file> --bin <bin_index> [--epoch <epoch>]
    
Example:
    python scripts/analyze_ghm_samples.py --metadata-file saved_models/metadata/metadata_v1.0.json --bin 0 --epoch 10
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Set, Any, Tuple
import wave
import pickle
import datetime

# Add the root directory to the Python path to ensure imports work correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from datasets.metadata import analyze_ghm_samples
from datasets import DatasetConfig

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze GHM sample distribution")
    
    parser.add_argument('--metadata-file', type=str, required=True,
                        help='Path to the dataset metadata file')
    parser.add_argument('--bin', type=int, required=True,
                        help='GHM bin index to analyze')
    parser.add_argument('--epoch', type=int, default=None,
                        help='Specific epoch to analyze (default: latest)')
    parser.add_argument('--output-dir', type=str, default='results/ghm_analysis',
                        help='Directory to save analysis results')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations of sample spectrograms')
    parser.add_argument('--list-only', action='store_true',
                        help='Only list samples without detailed analysis')
    parser.add_argument('--exclude', action='store_true',
                        help='Add found samples to exclusion list')
    parser.add_argument('--exclusion-file', type=str,
                        help='Path to exclusion file (required if --exclude is used)')
    
    args = parser.parse_args()
    
    if args.exclude and not args.exclusion_file:
        parser.error("--exclusion-file is required when using --exclude")
    
    return args

def get_sample_properties(metadata_file: str, samples: List[str]) -> List[Dict]:
    """
    Get properties of the specified samples from metadata.
    
    Args:
        metadata_file: Path to metadata file
        samples: List of sample IDs
        
    Returns:
        List of sample property dictionaries
    """
    properties = []
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    for sample_id in samples:
        if sample_id in metadata:
            props = metadata[sample_id]
            properties.append(props)
        else:
            print(f"Warning: Sample {sample_id} not found in metadata")
    
    return properties

def analyze_bin_samples(metadata_file: str, bin_idx: int, epoch: Optional[int] = None,
                       output_dir: str = 'results/ghm_analysis', visualize: bool = False) -> Dict:
    """
    Analyze samples assigned to a specific GHM bin.
    
    Args:
        metadata_file: Path to metadata file
        bin_idx: GHM bin index to analyze
        epoch: Specific epoch to analyze (None for latest)
        output_dir: Directory to save analysis results
        visualize: Whether to generate visualizations
        
    Returns:
        Dictionary with analysis results
    """
    # Get samples in the specified bin
    samples = analyze_ghm_samples(metadata_file, bin_idx, epoch)
    
    if not samples:
        print(f"No samples found in bin {bin_idx}" + (f" at epoch {epoch}" if epoch else ""))
        return {"samples": [], "stats": {}}
    
    print(f"Found {len(samples)} samples in bin {bin_idx}" + (f" at epoch {epoch}" if epoch else ""))
    
    # Get sample properties
    properties = get_sample_properties(metadata_file, samples)
    
    # Analyze distribution of properties
    stats = defaultdict(Counter)
    
    for prop in properties:
        for key, value in prop.items():
            if key not in ['id', 'file_path', 'ghm_bins', 'notes']:
                stats[key][value] += 1
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save analysis results
    result = {
        "bin_index": bin_idx,
        "epoch": epoch,
        "sample_count": len(samples),
        "samples": samples,
        "properties": properties,
        "statistics": {k: dict(v) for k, v in stats.items()}
    }
    
    epoch_str = f"epoch{epoch}_" if epoch else ""
    output_file = os.path.join(output_dir, f"bin{bin_idx}_{epoch_str}analysis.json")
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Analysis saved to {output_file}")
    
    # Generate visualizations if requested
    if visualize:
        vis_dir = os.path.join(output_dir, f"bin{bin_idx}_{epoch_str}visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        for key, counter in stats.items():
            plt.figure(figsize=(10, 6))
            labels = list(counter.keys())
            values = list(counter.values())
            
            plt.bar(labels, values)
            plt.title(f'Distribution of {key} in bin {bin_idx}')
            plt.savefig(os.path.join(vis_dir, f"{key}_distribution.png"))
            plt.close()
        
        print(f"Visualizations saved to {vis_dir}")
    
    return result

def add_to_exclusion_list(samples: List[str], exclusion_file: str) -> None:
    """
    Add samples to exclusion list.
    
    Args:
        samples: List of sample IDs to exclude
        exclusion_file: Path to exclusion file
    """
    # Load existing exclusions if the file exists
    exclusions = set()
    if os.path.exists(exclusion_file):
        with open(exclusion_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    exclusions.add(line)
    
    # Add new exclusions
    added = 0
    for sample in samples:
        if sample not in exclusions:
            exclusions.add(sample)
            added += 1
    
    # Save updated exclusion list
    with open(exclusion_file, 'w') as f:
        f.write("# Sample exclusion list - one sample ID per line\n")
        f.write(f"# Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        for sample in sorted(exclusions):
            f.write(f"{sample}\n")
    
    print(f"Added {added} new samples to exclusion list {exclusion_file}")
    print(f"Total excluded samples: {len(exclusions)}")

def main():
    """Main function to analyze GHM samples."""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.list_only:
        # Just list the samples without detailed analysis
        samples = analyze_ghm_samples(args.metadata_file, args.bin, args.epoch)
        if samples:
            print(f"Samples in bin {args.bin}" + (f" at epoch {args.epoch}" if args.epoch else ""))
            for i, sample in enumerate(samples):
                print(f"{i+1:3d}: {sample}")
        else:
            print(f"No samples found in bin {args.bin}" + (f" at epoch {args.epoch}" if args.epoch else ""))
    else:
        # Perform detailed analysis
        result = analyze_bin_samples(
            args.metadata_file, 
            args.bin, 
            args.epoch,
            args.output_dir,
            args.visualize
        )
        
        # Print summary statistics
        if result["statistics"]:
            print("\nSummary Statistics:")
            for key, counter in result["statistics"].items():
                print(f"\n{key.capitalize()} distribution:")
                for value, count in sorted(counter.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {value}: {count} samples ({count*100/result['sample_count']:.1f}%)")
    
    # Add to exclusion list if requested
    if args.exclude:
        samples = analyze_ghm_samples(args.metadata_file, args.bin, args.epoch)
        if samples:
            confirmation = input(f"Add {len(samples)} samples to exclusion list? (y/n): ")
            if confirmation.lower() == 'y':
                add_to_exclusion_list(samples, args.exclusion_file)
            else:
                print("Exclusion cancelled.")
        else:
            print("No samples to exclude.")

if __name__ == "__main__":
    main() 