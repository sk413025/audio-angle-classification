import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re

# Set the style
plt.style.use('ggplot')
sns.set_palette("viridis")

# Create directory for visualizations
os.makedirs('tracin_visualizations', exist_ok=True)

# Load the influence data
fp = '/Users/sbplab/Hank/sinetone_sliced/step_018_sliced/metadata/plastic_500hz_influence_metadata.json'
try:
    with open(fp, 'r') as f:
        data = json.load(f)
    print(f"Successfully loaded influence data from {fp}")
except FileNotFoundError:
    print(f"File not found: {fp}")
    print("Please run compute_tracin_influence.py with --compute-influence flag to generate test influence data")
    exit(1)
except json.JSONDecodeError:
    print(f"Error decoding JSON from {fp}")
    exit(1)

# Extract degree information
def extract_degrees(sample_id):
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

# Identify test pairs
test_pairs = set()
for train_pair_data in data.values():
    for key in train_pair_data.keys():
        if key.startswith('tracin_influence_') and not key.endswith('_per_checkpoint'):
            test_pairs.add(key.replace('tracin_influence_', ''))

test_pairs = sorted(list(test_pairs))
print(f"Found {len(test_pairs)} test pairs")

if not test_pairs:
    print("No test influence data found. Make sure to run compute_tracin_influence.py with --compute-influence")
    exit(0)

# 1. Analyze influence distribution for each test pair (limit to first 5)
plt.figure(figsize=(14, 10))
for i, test_pair in enumerate(test_pairs[:min(5, len(test_pairs))]):
    influences = []
    for train_pair_data in data.values():
        key = f'tracin_influence_{test_pair}'
        if key in train_pair_data:
            influences.append(train_pair_data[key])
    
    if influences:
        # Plot histogram of influence values
        plt.subplot(min(5, len(test_pairs)), 1, i+1)
        sns.histplot(influences, kde=True, bins=min(30, len(set(influences))))
        
        # Add vertical lines for notable statistics
        plt.axvline(np.mean(influences), color='red', linestyle='--', linewidth=1, label=f'Mean: {np.mean(influences):.2f}')
        plt.axvline(np.median(influences), color='green', linestyle='-.', linewidth=1, label=f'Median: {np.median(influences):.2f}')
        
        # Format and label
        first_deg, second_deg = extract_degrees(test_pair)
        plt.title(f'Influence Distribution for Test Pair: {first_deg}° → {second_deg}°')
        plt.xlabel('Influence Score')
        plt.ylabel('Frequency')
        plt.legend()

plt.tight_layout()
plt.savefig('tracin_visualizations/test_influence_distributions.png')
print("Saved test influence distributions plot")

# 2. Create a heatmap of average influence by degree pairs
degree_pairs = defaultdict(list)
for train_pair, train_data in data.items():
    train_first_deg, train_second_deg = extract_degrees(train_pair)
    
    for test_pair in test_pairs:
        key = f'tracin_influence_{test_pair}'
        if key in train_data:
            test_first_deg, test_second_deg = extract_degrees(test_pair)
            
            # Group by degrees
            pair_key = (train_first_deg, train_second_deg, test_first_deg, test_second_deg)
            degree_pairs[pair_key].append(train_data[key])

# Average the influence scores for each degree pair
avg_influence = {k: np.mean(v) for k, v in degree_pairs.items() if v}

# Get unique degrees
all_degrees = set()
for key in degree_pairs.keys():
    all_degrees.update([key[0], key[1], key[2], key[3]])
all_degrees = sorted(list(all_degrees))

if all_degrees and test_pairs:
    # Create matrices for the heatmaps
    for test_pair in test_pairs[:min(4, len(test_pairs))]:
        test_first_deg, test_second_deg = extract_degrees(test_pair)
        
        # Create a matrix for this test pair
        matrix = np.zeros((len(all_degrees), len(all_degrees)))
        count_matrix = np.zeros((len(all_degrees), len(all_degrees)))
        
        for (train_first_deg, train_second_deg, t_first_deg, t_second_deg), scores in degree_pairs.items():
            if t_first_deg == test_first_deg and t_second_deg == test_second_deg:
                if train_first_deg in all_degrees and train_second_deg in all_degrees:
                    i = all_degrees.index(train_first_deg)
                    j = all_degrees.index(train_second_deg)
                    matrix[i, j] = np.mean(scores)
                    count_matrix[i, j] = 1
        
        # Plot the heatmap if we have data
        if np.sum(count_matrix) > 0:
            plt.figure(figsize=(10, 8))
            mask = (count_matrix == 0)
            
            # Create diverging colormap with white at zero
            cmap = sns.diverging_palette(240, 10, as_cmap=True)
            
            sns.heatmap(
                matrix,
                annot=True,
                fmt=".1f",
                mask=mask,
                cmap=cmap,
                center=0,
                xticklabels=[f"{d}°" for d in all_degrees],
                yticklabels=[f"{d}°" for d in all_degrees]
            )
            
            plt.title(f'Influence of Training Pairs on Test Pair: {test_first_deg}° → {test_second_deg}°')
            plt.xlabel('Training Second Degree')
            plt.ylabel('Training First Degree')
            plt.tight_layout()
            
            plt.savefig(f'tracin_visualizations/test_influence_heatmap_{test_first_deg}_{test_second_deg}.png')
            print(f"Saved influence heatmap for test pair {test_first_deg}° → {test_second_deg}°")

# 3. Identify most influential training pairs for each test pair
for test_pair in test_pairs[:min(5, len(test_pairs))]:
    # Collect influence scores for this test pair
    pair_influences = {}
    for train_pair, train_data in data.items():
        key = f'tracin_influence_{test_pair}'
        if key in train_data:
            pair_influences[train_pair] = train_data[key]
    
    if pair_influences:
        # Sort and get top and bottom 5
        sorted_influence = sorted(pair_influences.items(), key=lambda x: x[1], reverse=True)
        top_pairs = sorted_influence[:5]
        bottom_pairs = sorted_influence[-5:]
        
        plt.figure(figsize=(12, 8))
        
        # Plot top pairs
        plt.subplot(2, 1, 1)
        top_labels = [f"{extract_degrees(p[0])[0]}°→{extract_degrees(p[0])[1]}°" for p in top_pairs]
        top_values = [p[1] for p in top_pairs]
        colors = ['green' if v > 0 else 'red' for v in top_values]
        plt.barh(range(len(top_labels)), top_values, color=colors)
        plt.yticks(range(len(top_labels)), top_labels)
        test_first_deg, test_second_deg = extract_degrees(test_pair)
        plt.title(f'Top 5 Most Influential Training Pairs on Test Pair {test_first_deg}°→{test_second_deg}°')
        plt.xlabel('Influence Score')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Plot bottom pairs
        plt.subplot(2, 1, 2)
        bottom_labels = [f"{extract_degrees(p[0])[0]}°→{extract_degrees(p[0])[1]}°" for p in bottom_pairs]
        bottom_values = [p[1] for p in bottom_pairs]
        colors = ['green' if v > 0 else 'red' for v in bottom_values]
        plt.barh(range(len(bottom_labels)), bottom_values, color=colors)
        plt.yticks(range(len(bottom_labels)), bottom_labels)
        plt.title(f'Bottom 5 Least Influential Training Pairs on Test Pair {test_first_deg}°→{test_second_deg}°')
        plt.xlabel('Influence Score')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'tracin_visualizations/test_influence_top_bottom_{test_first_deg}_{test_second_deg}.png')
        print(f"Saved top/bottom influential pairs for test pair {test_first_deg}°→{test_second_deg}°")

# 4. Analyze influence by angle difference
angle_diff_data = defaultdict(list)
for train_pair, train_data in data.items():
    train_first_deg, train_second_deg = extract_degrees(train_pair)
    train_diff = min(abs(train_first_deg - train_second_deg), 360 - abs(train_first_deg - train_second_deg))
    
    for test_pair in test_pairs:
        key = f'tracin_influence_{test_pair}'
        if key in train_data:
            test_first_deg, test_second_deg = extract_degrees(test_pair)
            test_diff = min(abs(test_first_deg - test_second_deg), 360 - abs(test_first_deg - test_second_deg))
            
            # Analyze if similar angle differences have similar influence
            angle_key = (train_diff, test_diff)
            angle_diff_data[angle_key].append(train_data[key])

# Calculate statistics for each angle difference pair
angle_stats = {}
for angle_key, scores in angle_diff_data.items():
    if scores:
        angle_stats[angle_key] = {
            'mean': np.mean(scores),
            'median': np.median(scores),
            'std': np.std(scores),
            'count': len(scores)
        }

# Plot influence by angle difference similarity
if angle_stats:
    # Group by test angle difference
    test_angles = sorted(set([k[1] for k in angle_stats.keys()]))
    
    plt.figure(figsize=(12, 8))
    
    for i, test_angle in enumerate(test_angles):
        train_angles = []
        means = []
        stds = []
        
        for (train_diff, test_diff), stats in angle_stats.items():
            if test_diff == test_angle:
                train_angles.append(train_diff)
                means.append(stats['mean'])
                stds.append(stats['std'])
        
        # Sort by training angle difference
        if train_angles:
            sort_idx = np.argsort(train_angles)
            train_angles = [train_angles[i] for i in sort_idx]
            means = [means[i] for i in sort_idx]
            stds = [stds[i] for i in sort_idx]
            
            plt.errorbar(
                train_angles, 
                means, 
                yerr=stds, 
                marker='o', 
                linestyle='-', 
                label=f'Test Angle Diff: {test_angle}°',
                capsize=4
            )
    
    plt.xlabel('Training Angle Difference (degrees)')
    plt.ylabel('Average Influence Score')
    plt.title('Influence by Angle Difference Similarity')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('tracin_visualizations/influence_by_angle_similarity.png')
    print("Saved influence by angle similarity analysis")

# 5. Create a correlation matrix between training and test angle differences
if test_angles and all_degrees:
    corr_matrix = np.zeros((len(all_degrees), len(all_degrees)))
    count_matrix = np.zeros((len(all_degrees), len(all_degrees)))
    
    # Compute correlation for each angle pair
    for test_first, test_second in [(d1, d2) for d1 in all_degrees for d2 in all_degrees if d1 != d2]:
        # Get all influence scores for this test pair
        test_scores = []
        train_diffs = []
        
        for train_pair, train_data in data.items():
            key = f'tracin_influence_{test_first}_{test_second}'
            alt_key = f'tracin_influence_{test_first}_{test_second}_val'
            if key in train_data:
                train_first_deg, train_second_deg = extract_degrees(train_pair)
                train_diff = min(abs(train_first_deg - train_second_deg), 360 - abs(train_first_deg - train_second_deg))
                
                test_scores.append(train_data[key])
                train_diffs.append(train_diff)
            elif alt_key in train_data:
                train_first_deg, train_second_deg = extract_degrees(train_pair)
                train_diff = min(abs(train_first_deg - train_second_deg), 360 - abs(train_first_deg - train_second_deg))
                
                test_scores.append(train_data[alt_key])
                train_diffs.append(train_diff)
        
        # Compute correlation if we have enough data
        if len(test_scores) > 5 and len(set(train_diffs)) > 1:
            corr = np.corrcoef(train_diffs, test_scores)[0, 1]
            
            # Store in matrix
            i = all_degrees.index(test_first)
            j = all_degrees.index(test_second)
            corr_matrix[i, j] = corr
            count_matrix[i, j] = 1
    
    # Plot correlation matrix
    if np.sum(count_matrix) > 0:
        plt.figure(figsize=(10, 8))
        mask = (count_matrix == 0)
        
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            mask=mask,
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            xticklabels=[f"{d}°" for d in all_degrees],
            yticklabels=[f"{d}°" for d in all_degrees]
        )
        
        plt.title('Correlation Between Training Angle Difference and Test Pair Influence')
        plt.xlabel('Second Degree in Test Pair')
        plt.ylabel('First Degree in Test Pair')
        plt.tight_layout()
        plt.savefig('tracin_visualizations/angle_correlation_matrix.png')
        print("Saved angle correlation matrix")

print("\nAll test influence visualizations have been saved to the 'tracin_visualizations' directory.") 