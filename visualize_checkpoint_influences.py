import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import defaultdict
from pathlib import Path

# Set the style
plt.style.use('ggplot')
sns.set_palette("viridis")

# Create directory for visualizations
os.makedirs('tracin_visualizations/checkpoints', exist_ok=True)

def extract_degrees(sample_id):
    """Extract degree information from sample ID."""
    try:
        # Use regex to find all occurrences of deg followed by numbers
        deg_matches = re.findall(r'deg(\d+)', sample_id)
        
        if len(deg_matches) >= 2:
            return int(deg_matches[0]), int(deg_matches[1])
        elif len(deg_matches) == 1:
            return int(deg_matches[0]), 0
        else:
            # If no deg pattern is found, try to extract numbers directly
            num_matches = re.findall(r'(\d+)(?:hz|_)', sample_id.lower())
            filtered_matches = [int(m) for m in num_matches if int(m) in [0, 36, 72, 108, 144, 180]]
            
            if len(filtered_matches) >= 2:
                return filtered_matches[0], filtered_matches[1]
            elif len(filtered_matches) == 1:
                return filtered_matches[0], 0
            else:
                print(f"Warning: Could not extract degrees from {sample_id}")
                return 0, 0
    except Exception as e:
        print(f"Error extracting degrees from {sample_id}: {e}")
        return 0, 0

def extract_checkpoint_number(checkpoint_key):
    """Extract checkpoint number from the key."""
    # Match model_epoch_X.pt pattern
    match = re.search(r'model_epoch_(\d+)\.pt', checkpoint_key)
    if match:
        return int(match.group(1))
    
    # Fallback to cp pattern
    match = re.search(r'_cp(\d+)$', checkpoint_key)
    if match:
        return int(match.group(1))
    
    return 0

def load_checkpoint_data(material='plastic', frequency='500hz'):
    """Load per-checkpoint influence data from metadata file."""
    metadata_path = f'/Users/sbplab/Hank/sinetone_sliced/step_018_sliced/metadata/{material}_{frequency}_per_checkpoint_influence_metadata.json'
    
    try:
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded per-checkpoint influence data from {metadata_path}")
        return data
    except FileNotFoundError:
        print(f"File not found: {metadata_path}")
        print("Please run compute_tracin_influence.py with --save-per-checkpoint-influence flag to generate per-checkpoint influence data")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {metadata_path}")
        return None

def identify_test_pairs(data):
    """Identify all test pairs for which we have per-checkpoint influence scores."""
    test_pairs = set()
    
    # Pattern to match model_epoch format
    pattern = re.compile(r'tracin_influence_(.+?)_model_epoch')
    
    for train_pair_data in data.values():
        for key in train_pair_data.keys():
            if 'tracin_influence_' in key and 'model_epoch' in key:
                match = pattern.search(key)
                if match:
                    test_pairs.add(match.group(1))
    
    return sorted(list(test_pairs))

def plot_influence_progression(data, test_pair, output_dir='tracin_visualizations/checkpoints'):
    """
    Plot the progression of influence scores across checkpoints for a test pair.
    
    Args:
        data: The per-checkpoint influence data dictionary
        test_pair: The test pair ID to plot for
        output_dir: Directory to save the plots
    """
    # Extract test degrees
    test_first_deg, test_second_deg = extract_degrees(test_pair)
    
    # Create a dictionary to track influence for each training pair across checkpoints
    progression_data = defaultdict(dict)
    
    # Identify all checkpoints
    all_checkpoints = set()
    
    # Collect progression data
    for train_pair, train_data in data.items():
        for key, score in train_data.items():
            if f'tracin_influence_{test_pair}_model_epoch' in key:
                checkpoint = extract_checkpoint_number(key)
                progression_data[train_pair][checkpoint] = score
                all_checkpoints.add(checkpoint)
    
    if not progression_data:
        print(f"No per-checkpoint influence data found for test pair {test_pair}")
        return
    
    # Sort checkpoints
    all_checkpoints = sorted(list(all_checkpoints))
    
    # Plot 1: Overall influence progression across checkpoints (aggregated)
    plt.figure(figsize=(14, 8))
    
    # Compute average/median influence across all training pairs per checkpoint
    avg_influences = []
    median_influences = []
    checkpoint_labels = []
    
    for cp in all_checkpoints:
        cp_scores = [progression_data[train_pair].get(cp, 0) for train_pair in progression_data if cp in progression_data[train_pair]]
        if cp_scores:
            avg_influences.append(np.mean(cp_scores))
            median_influences.append(np.median(cp_scores))
            checkpoint_labels.append(f"Epoch {cp}")
    
    # Plot averages
    plt.plot(checkpoint_labels, avg_influences, marker='o', linestyle='-', linewidth=2, label='Mean Influence')
    plt.plot(checkpoint_labels, median_influences, marker='s', linestyle='--', linewidth=2, label='Median Influence')
    
    # Add annotations for significant checkpoints (e.g., min/max)
    min_idx = np.argmin(avg_influences)
    max_idx = np.argmax(avg_influences)
    
    plt.annotate(f"Min: {avg_influences[min_idx]:.2f}", 
                xy=(min_idx, avg_influences[min_idx]),
                xytext=(min_idx, avg_influences[min_idx] - 0.5),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=10)
    
    plt.annotate(f"Max: {avg_influences[max_idx]:.2f}", 
                xy=(max_idx, avg_influences[max_idx]),
                xytext=(max_idx, avg_influences[max_idx] + 0.5),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=10)
    
    plt.title(f'Influence Progression Across Checkpoints for Test Pair {test_first_deg}°→{test_second_deg}°')
    plt.xlabel('Checkpoint')
    plt.ylabel('Average Influence Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f'influence_progression_{test_first_deg}_{test_second_deg}.png'))
    print(f"Saved influence progression plot for test pair {test_first_deg}°→{test_second_deg}°")
    
    # Plot 2: Influence progression for top 5 most influential training pairs
    plt.figure(figsize=(14, 8))
    
    # Find the top 5 most influential training pairs based on final checkpoint
    final_checkpoint = max(all_checkpoints)
    final_influences = {}
    for train_pair in progression_data:
        if final_checkpoint in progression_data[train_pair]:
            final_influences[train_pair] = progression_data[train_pair][final_checkpoint]
    
    # Sort and get top 5
    top_train_pairs = sorted(final_influences.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    
    # Plot each top pair's progression
    for train_pair, _ in top_train_pairs:
        train_first_deg, train_second_deg = extract_degrees(train_pair)
        label = f"{train_first_deg}°→{train_second_deg}°"
        
        checkpoints = sorted(progression_data[train_pair].keys())
        influences = [progression_data[train_pair][cp] for cp in checkpoints]
        
        plt.plot([f"Epoch {cp}" for cp in checkpoints], influences, marker='o', linestyle='-', label=label)
    
    plt.title(f'Influence Progression for Top 5 Most Influential Training Pairs on Test Pair {test_first_deg}°→{test_second_deg}°')
    plt.xlabel('Checkpoint')
    plt.ylabel('Influence Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f'top5_progression_{test_first_deg}_{test_second_deg}.png'))
    print(f"Saved top 5 influence progression plot for test pair {test_first_deg}°→{test_second_deg}°")
    
    # Plot 3: Heatmap comparison of first vs last checkpoint
    if len(all_checkpoints) >= 2:
        first_cp = min(all_checkpoints)
        last_cp = max(all_checkpoints)
        
        # Collect data for each train degree pair
        first_cp_data = defaultdict(float)
        last_cp_data = defaultdict(float)
        
        for train_pair in progression_data:
            if first_cp in progression_data[train_pair] and last_cp in progression_data[train_pair]:
                train_first_deg, train_second_deg = extract_degrees(train_pair)
                pair_key = (train_first_deg, train_second_deg)
                
                first_cp_data[pair_key] += progression_data[train_pair][first_cp]
                last_cp_data[pair_key] += progression_data[train_pair][last_cp]
        
        # Get all unique degrees
        all_degrees = sorted(set([deg for pair in list(first_cp_data.keys()) + list(last_cp_data.keys()) for deg in pair]))
        
        if all_degrees:
            # Create matrices for the heatmaps
            first_matrix = np.zeros((len(all_degrees), len(all_degrees)))
            last_matrix = np.zeros((len(all_degrees), len(all_degrees)))
            diff_matrix = np.zeros((len(all_degrees), len(all_degrees)))
            
            # Fill matrices
            for (first_deg, second_deg), score in first_cp_data.items():
                if first_deg in all_degrees and second_deg in all_degrees:
                    i = all_degrees.index(first_deg)
                    j = all_degrees.index(second_deg)
                    first_matrix[i, j] = score
            
            for (first_deg, second_deg), score in last_cp_data.items():
                if first_deg in all_degrees and second_deg in all_degrees:
                    i = all_degrees.index(first_deg)
                    j = all_degrees.index(second_deg)
                    last_matrix[i, j] = score
                    diff_matrix[i, j] = score - first_matrix[i, j]
            
            # Create a figure with 3 subplots
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Create diverging colormap with white at zero
            cmap = sns.diverging_palette(240, 10, as_cmap=True)
            
            # First checkpoint heatmap
            im1 = sns.heatmap(
                first_matrix,
                annot=True,
                fmt=".1f",
                cmap=cmap,
                center=0,
                xticklabels=[f"{d}°" for d in all_degrees],
                yticklabels=[f"{d}°" for d in all_degrees],
                ax=axes[0]
            )
            axes[0].set_title(f'First Checkpoint (Epoch {first_cp})')
            axes[0].set_xlabel('Training Second Degree')
            axes[0].set_ylabel('Training First Degree')
            
            # Last checkpoint heatmap
            im2 = sns.heatmap(
                last_matrix,
                annot=True,
                fmt=".1f",
                cmap=cmap,
                center=0,
                xticklabels=[f"{d}°" for d in all_degrees],
                yticklabels=[f"{d}°" for d in all_degrees],
                ax=axes[1]
            )
            axes[1].set_title(f'Last Checkpoint (Epoch {last_cp})')
            axes[1].set_xlabel('Training Second Degree')
            axes[1].set_ylabel('Training First Degree')
            
            # Difference heatmap
            im3 = sns.heatmap(
                diff_matrix,
                annot=True,
                fmt=".1f",
                cmap=cmap,
                center=0,
                xticklabels=[f"{d}°" for d in all_degrees],
                yticklabels=[f"{d}°" for d in all_degrees],
                ax=axes[2]
            )
            axes[2].set_title('Difference (Last - First)')
            axes[2].set_xlabel('Training Second Degree')
            axes[2].set_ylabel('Training First Degree')
            
            plt.suptitle(f'Influence Change from First to Last Checkpoint for Test Pair {test_first_deg}°→{test_second_deg}°')
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, f'checkpoint_comparison_{test_first_deg}_{test_second_deg}.png'))
            print(f"Saved checkpoint comparison heatmaps for test pair {test_first_deg}°→{test_second_deg}°")

def plot_convergence_analysis(data, test_pairs, output_dir='tracin_visualizations/checkpoints'):
    """
    Analyze how quickly influence scores converge across checkpoints.
    
    Args:
        data: The per-checkpoint influence data dictionary
        test_pairs: List of test pairs to analyze
        output_dir: Directory to save the plots
    """
    # Extract all checkpoint numbers
    all_checkpoints = set()
    for train_pair_data in data.values():
        for key in train_pair_data.keys():
            if 'model_epoch' in key:
                cp = extract_checkpoint_number(key)
                all_checkpoints.add(cp)
    
    all_checkpoints = sorted(list(all_checkpoints))
    
    if len(all_checkpoints) < 3:
        print("Not enough checkpoints for convergence analysis")
        return
    
    # For each test pair, compute variance reduction across checkpoints
    plt.figure(figsize=(14, 8))
    
    legend_entries = []
    
    for test_pair in test_pairs[:min(5, len(test_pairs))]:
        test_first_deg, test_second_deg = extract_degrees(test_pair)
        legend_entry = f"{test_first_deg}°→{test_second_deg}°"
        legend_entries.append(legend_entry)
        
        # For each checkpoint, compute variance of influence scores
        variances = []
        
        for cp in all_checkpoints:
            cp_scores = []
            for train_pair, train_data in data.items():
                key_pattern = f'tracin_influence_{test_pair}_model_epoch_{cp}.pt'
                for key in train_data.keys():
                    if key_pattern in key:
                        cp_scores.append(train_data[key])
            
            if cp_scores:
                variances.append(np.var(cp_scores))
            else:
                variances.append(0)
        
        # Normalize variances
        if max(variances) > 0:
            variances = [v / max(variances) for v in variances]
        
        # Plot variance reduction
        plt.plot(all_checkpoints, variances, marker='o', linestyle='-', label=legend_entry)
    
    plt.title('Convergence Analysis: Normalized Variance of Influence Scores Across Checkpoints')
    plt.xlabel('Checkpoint Epoch')
    plt.ylabel('Normalized Variance')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'influence_convergence.png'))
    print("Saved influence convergence analysis plot")

def main():
    # Load the per-checkpoint influence data
    data = load_checkpoint_data(material='plastic', frequency='500hz')
    
    if not data:
        print("No data available. Exiting.")
        return
    
    # Identify test pairs with per-checkpoint influence data
    test_pairs = identify_test_pairs(data)
    
    if not test_pairs:
        print("No test pairs found with per-checkpoint influence data.")
        return
    
    print(f"Found {len(test_pairs)} test pairs with per-checkpoint influence data")
    
    # For each test pair, plot influence progression
    for i, test_pair in enumerate(test_pairs[:min(5, len(test_pairs))]):  # Limit to 5 pairs for brevity
        print(f"Processing test pair {i+1}/{min(5, len(test_pairs))}: {test_pair}")
        plot_influence_progression(data, test_pair)
    
    # Plot convergence analysis across test pairs
    plot_convergence_analysis(data, test_pairs)
    
    print("Visualization of checkpoint influences completed")

if __name__ == "__main__":
    main() 