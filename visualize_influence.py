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
fp = '/Users/sbplab/Hank/sinetone_sliced/step_018_sliced/metadata/plastic_500hz_metadata.json'
with open(fp, 'r') as f:
    data = json.load(f)

# Extract degree information
def extract_degrees(sample_id):
    try:
        # Use regex to find all occurrences of deg followed by numbers
        deg_matches = re.findall(r'deg(\d+)', sample_id)
        
        if len(deg_matches) >= 2:
            return int(deg_matches[0]), int(deg_matches[1])
        elif len(deg_matches) == 1:
            print(f"Warning: Only found one degree in {sample_id}")
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

# Extract self-influence scores
self_influence_scores = {}
for pair_id, pair_data in data.items():
    if 'tracin_self_influence' in pair_data:
        self_influence_scores[pair_id] = pair_data['tracin_self_influence']

print(f"Found {len(self_influence_scores)} samples with self-influence scores")

# Group data by degree pairs
degree_pairs = defaultdict(list)
for pair_id, score in self_influence_scores.items():
    first_deg, second_deg = extract_degrees(pair_id)
    degree_key = f"{first_deg}° vs {second_deg}°"
    degree_pairs[degree_key].append(score)

# Calculate statistics for each degree pair
degree_stats = {}
for degree_key, scores in degree_pairs.items():
    if scores:  # Ensure we have scores for this pair
        degree_stats[degree_key] = {
            'mean': np.mean(scores),
            'median': np.median(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'count': len(scores)
        }

# Sort degree pairs by mean influence score
sorted_pairs = sorted(degree_stats.items(), key=lambda x: x[1]['mean'], reverse=True)

if sorted_pairs:
    # 1. Create a bar chart of average self-influence by degree pair
    plt.figure(figsize=(12, 8))
    pairs = [pair[0] for pair in sorted_pairs]
    means = [pair[1]['mean'] for pair in sorted_pairs]
    stds = [pair[1]['std'] for pair in sorted_pairs]

    plt.bar(range(len(pairs)), means, yerr=stds, capsize=4, alpha=0.7)
    plt.xticks(range(len(pairs)), pairs, rotation=90)
    plt.ylabel('Average Self-Influence Score')
    plt.title('Self-Influence Scores by Degree Pair')
    plt.tight_layout()
    plt.savefig('tracin_visualizations/self_influence_by_pair.png')
    print(f"Saved self-influence by pair chart to tracin_visualizations/self_influence_by_pair.png")

    # 2. Create a heatmap of average self-influence by degree
    try:
        # Extract unique degrees from the pairs
        degrees = set()
        for key in degree_pairs.keys():
            match1 = re.search(r'(\d+)°', key.split(' vs ')[0])
            match2 = re.search(r'(\d+)°', key.split(' vs ')[1])
            if match1 and match2:
                degrees.add(int(match1.group(1)))
                degrees.add(int(match2.group(1)))
        
        all_degrees = sorted(list(degrees))
        
        if all_degrees:
            # Initialize the heatmap matrix
            heatmap_data = np.zeros((len(all_degrees), len(all_degrees)))
            count_matrix = np.zeros((len(all_degrees), len(all_degrees)))
            
            # Fill the heatmap matrix
            for pair_id, score in self_influence_scores.items():
                first_deg, second_deg = extract_degrees(pair_id)
                
                # Skip if degrees are not in our expected list
                if first_deg not in all_degrees or second_deg not in all_degrees:
                    continue
                
                # Find the indices in all_degrees
                i = all_degrees.index(first_deg)
                j = all_degrees.index(second_deg)
                
                # Update the matrices
                heatmap_data[i, j] += score
                count_matrix[i, j] += 1
            
            # Check if we have data to create a heatmap
            if np.sum(count_matrix) > 0:
                # Compute the average for each cell (handle division by zero)
                with np.errstate(divide='ignore', invalid='ignore'):
                    avg_heatmap = np.divide(heatmap_data, count_matrix)
                    avg_heatmap = np.nan_to_num(avg_heatmap, nan=0)
                
                # Create the heatmap
                plt.figure(figsize=(10, 8))
                mask = (count_matrix == 0)  # Create mask for cells with no data
                sns.heatmap(avg_heatmap, annot=True, fmt=".1f", cmap="YlGnBu", 
                            xticklabels=[f"{d}°" for d in all_degrees], 
                            yticklabels=[f"{d}°" for d in all_degrees],
                            mask=mask)
                plt.title('Average Self-Influence Score by Degree Combination')
                plt.xlabel('Second Degree')
                plt.ylabel('First Degree')
                plt.tight_layout()
                plt.savefig('tracin_visualizations/self_influence_heatmap.png')
                print(f"Saved self-influence heatmap to tracin_visualizations/self_influence_heatmap.png")
            else:
                print("Not enough data for heatmap visualization")
    except Exception as e:
        print(f"Error creating heatmap: {e}")

    # 3. Distribution of self-influence scores
    if self_influence_scores:
        plt.figure(figsize=(10, 6))
        all_scores = list(self_influence_scores.values())
        plt.hist(all_scores, bins=min(30, len(set(all_scores))), alpha=0.7)
        plt.axvline(np.mean(all_scores), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(all_scores):.2f}')
        plt.axvline(np.median(all_scores), color='green', linestyle='dashed', linewidth=2, label=f'Median: {np.median(all_scores):.2f}')
        plt.xlabel('Self-Influence Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Self-Influence Scores')
        plt.legend()
        plt.tight_layout()
        plt.savefig('tracin_visualizations/self_influence_distribution.png')
        print(f"Saved self-influence distribution to tracin_visualizations/self_influence_distribution.png")

    # 4. Find and visualize the most influential pairs
    if len(self_influence_scores) >= 2:  # Need at least two pairs for top/bottom
        top_n = min(10, len(self_influence_scores) // 2)  # Adjust top_n based on available data
        sorted_scores = sorted(self_influence_scores.items(), key=lambda x: x[1], reverse=True)
        top_pairs = sorted_scores[:top_n]
        bottom_pairs = sorted_scores[-top_n:]

        plt.figure(figsize=(12, 8))

        # Plot top pairs
        plt.subplot(2, 1, 1)
        plt.barh([p[0] for p in reversed(top_pairs)], [p[1] for p in reversed(top_pairs)])
        plt.title(f'Top {top_n} Most Influential Pairs')
        plt.xlabel('Self-Influence Score')
        plt.tight_layout()

        # Plot bottom pairs
        plt.subplot(2, 1, 2)
        plt.barh([p[0] for p in bottom_pairs], [p[1] for p in bottom_pairs])
        plt.title(f'Bottom {top_n} Least Influential Pairs')
        plt.xlabel('Self-Influence Score')
        plt.tight_layout()

        plt.subplots_adjust(hspace=0.5)
        plt.savefig('tracin_visualizations/top_bottom_influential_pairs.png')
        print(f"Saved top/bottom influential pairs to tracin_visualizations/top_bottom_influential_pairs.png")

    # 5. Visualize influence by angle difference
    angle_diff_data = defaultdict(list)
    for pair_id, score in self_influence_scores.items():
        first_deg, second_deg = extract_degrees(pair_id)
        # Calculate angle difference
        angle_diff = min(abs(first_deg - second_deg), 360 - abs(first_deg - second_deg))
        angle_diff_data[angle_diff].append(score)

    # Calculate statistics for each angle difference (if we have data)
    if angle_diff_data:
        angle_stats = {}
        for angle_diff, scores in angle_diff_data.items():
            if scores:  # Ensure we have scores
                angle_stats[angle_diff] = {
                    'mean': np.mean(scores),
                    'median': np.median(scores),
                    'std': np.std(scores),
                    'count': len(scores)
                }

        # Plot influence by angle difference (if we have stats)
        if angle_stats:
            plt.figure(figsize=(10, 6))
            angles = sorted(angle_stats.keys())
            means = [angle_stats[a]['mean'] for a in angles]
            stds = [angle_stats[a]['std'] for a in angles]

            plt.errorbar(angles, means, yerr=stds, marker='o', linestyle='-', capsize=4)
            plt.xlabel('Angle Difference (degrees)')
            plt.ylabel('Average Self-Influence Score')
            plt.title('Self-Influence Score vs. Angle Difference')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig('tracin_visualizations/influence_by_angle_diff.png')
            print(f"Saved influence by angle difference to tracin_visualizations/influence_by_angle_diff.png")

    # Print summary statistics
    if self_influence_scores:
        print("\nSummary Statistics:")
        print(f"Total pairs analyzed: {len(self_influence_scores)}")
        print(f"Average self-influence across all pairs: {np.mean(list(self_influence_scores.values())):.2f}")
        print(f"Median self-influence: {np.median(list(self_influence_scores.values())):.2f}")
        print(f"Range of self-influence: {np.min(list(self_influence_scores.values())):.2f} to {np.max(list(self_influence_scores.values())):.2f}")

    print("\nAll visualizations have been saved to the 'tracin_visualizations' directory.")
else:
    print("No data available for visualization. Check if self-influence scores exist in the metadata file.") 