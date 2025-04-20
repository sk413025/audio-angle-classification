import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import re
from collections import defaultdict

# Set the style
plt.style.use('ggplot')
sns.set_palette("viridis")
sns.set_context("paper", font_scale=1.2)

# Create directory for the report
os.makedirs('tracin_reports', exist_ok=True)

# Paths to metadata files
self_influence_file = '/Users/sbplab/Hank/sinetone_sliced/step_018_sliced/metadata/plastic_500hz_metadata.json'
test_influence_file = '/Users/sbplab/Hank/sinetone_sliced/step_018_sliced/metadata/plastic_500hz_influence_metadata.json'

# Load metadata files
print("Loading metadata files...")
self_influence_data = {}
test_influence_data = {}

try:
    with open(self_influence_file, 'r') as f:
        self_influence_data = json.load(f)
    print(f"Successfully loaded self-influence data from {self_influence_file}")
except FileNotFoundError:
    print(f"Warning: Self-influence file not found: {self_influence_file}")
except json.JSONDecodeError:
    print(f"Warning: Error decoding JSON from {self_influence_file}")

try:
    with open(test_influence_file, 'r') as f:
        test_influence_data = json.load(f)
    print(f"Successfully loaded test influence data from {test_influence_file}")
except FileNotFoundError:
    print(f"Warning: Test influence file not found: {test_influence_file}")
except json.JSONDecodeError:
    print(f"Warning: Error decoding JSON from {test_influence_file}")

if not self_influence_data and not test_influence_data:
    print("Error: No data available to generate report.")
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
                # Silent failure for report generation
                return 0, 0
    except Exception:
        return 0, 0

# Calculate angle difference
def angle_difference(angle1, angle2):
    return min(abs(angle1 - angle2), 360 - abs(angle1 - angle2))

# Create PDF report
report_filename = f'tracin_reports/tracin_influence_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
with PdfPages(report_filename) as pdf:
    # Add a title page
    plt.figure(figsize=(12, 9))
    plt.axis('off')
    plt.text(0.5, 0.7, 'TracIn Influence Analysis Report', 
             horizontalalignment='center', fontsize=30, weight='bold')
    plt.text(0.5, 0.6, 'Plastic 500Hz Dataset', 
             horizontalalignment='center', fontsize=24)
    plt.text(0.5, 0.5, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
             horizontalalignment='center', fontsize=16)
    plt.text(0.5, 0.4, 'Based on "Estimating Training Data Influence\nby Tracing Gradient Descent"', 
             horizontalalignment='center', fontsize=14)
    pdf.savefig()
    plt.close()
    
    # ---- Self-Influence Analysis ----
    if self_influence_data:
        # Extract self-influence scores
        self_influence_scores = {}
        for pair_id, pair_data in self_influence_data.items():
            if 'tracin_self_influence' in pair_data:
                self_influence_scores[pair_id] = pair_data['tracin_self_influence']
        
        if self_influence_scores:
            # 1. Section header for Self-Influence
            plt.figure(figsize=(12, 9))
            plt.axis('off')
            plt.text(0.5, 0.5, 'Self-Influence Analysis', 
                     horizontalalignment='center', fontsize=24, weight='bold')
            plt.text(0.5, 0.4, f'Total Samples: {len(self_influence_scores)}', 
                     horizontalalignment='center', fontsize=16)
            pdf.savefig()
            plt.close()
            
            # 2. Self-influence score distribution
            plt.figure(figsize=(10, 6))
            all_scores = list(self_influence_scores.values())
            plt.hist(all_scores, bins=min(30, len(set(all_scores))), alpha=0.7)
            plt.axvline(np.mean(all_scores), color='red', linestyle='dashed', linewidth=2, 
                        label=f'Mean: {np.mean(all_scores):.2f}')
            plt.axvline(np.median(all_scores), color='green', linestyle='dashed', linewidth=2, 
                        label=f'Median: {np.median(all_scores):.2f}')
            plt.xlabel('Self-Influence Score')
            plt.ylabel('Frequency')
            plt.title('Distribution of Self-Influence Scores')
            plt.legend()
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            
            # 3. Group data by degree pairs
            degree_pairs = defaultdict(list)
            for pair_id, score in self_influence_scores.items():
                first_deg, second_deg = extract_degrees(pair_id)
                degree_key = f"{first_deg}° vs {second_deg}°"
                degree_pairs[degree_key].append(score)
            
            degree_stats = {}
            for degree_key, scores in degree_pairs.items():
                if scores:
                    degree_stats[degree_key] = {
                        'mean': np.mean(scores),
                        'median': np.median(scores),
                        'std': np.std(scores),
                        'min': np.min(scores),
                        'max': np.max(scores),
                        'count': len(scores)
                    }
            
            # 4. Bar chart of average self-influence by degree pair
            sorted_pairs = sorted(degree_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
            
            if sorted_pairs:
                plt.figure(figsize=(12, 8))
                pairs = [pair[0] for pair in sorted_pairs]
                means = [pair[1]['mean'] for pair in sorted_pairs]
                stds = [pair[1]['std'] for pair in sorted_pairs]
                
                plt.bar(range(len(pairs)), means, yerr=stds, capsize=4, alpha=0.7)
                plt.xticks(range(len(pairs)), pairs, rotation=90)
                plt.ylabel('Average Self-Influence Score')
                plt.title('Self-Influence Scores by Degree Pair')
                plt.tight_layout()
                pdf.savefig()
                plt.close()
            
            # 5. Heatmap of average self-influence by degree
            degrees = set()
            for key in degree_pairs.keys():
                match1 = re.search(r'(\d+)°', key.split(' vs ')[0])
                match2 = re.search(r'(\d+)°', key.split(' vs ')[1])
                if match1 and match2:
                    degrees.add(int(match1.group(1)))
                    degrees.add(int(match2.group(1)))
            
            all_degrees = sorted(list(degrees))
            
            if all_degrees:
                heatmap_data = np.zeros((len(all_degrees), len(all_degrees)))
                count_matrix = np.zeros((len(all_degrees), len(all_degrees)))
                
                for pair_id, score in self_influence_scores.items():
                    first_deg, second_deg = extract_degrees(pair_id)
                    
                    if first_deg in all_degrees and second_deg in all_degrees:
                        i = all_degrees.index(first_deg)
                        j = all_degrees.index(second_deg)
                        heatmap_data[i, j] += score
                        count_matrix[i, j] += 1
                
                if np.sum(count_matrix) > 0:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        avg_heatmap = np.divide(heatmap_data, count_matrix)
                        avg_heatmap = np.nan_to_num(avg_heatmap, nan=0)
                    
                    plt.figure(figsize=(10, 8))
                    mask = (count_matrix == 0)
                    sns.heatmap(avg_heatmap, annot=True, fmt=".1f", cmap="YlGnBu", 
                                xticklabels=[f"{d}°" for d in all_degrees], 
                                yticklabels=[f"{d}°" for d in all_degrees],
                                mask=mask)
                    plt.title('Average Self-Influence Score by Degree Combination')
                    plt.xlabel('Second Degree')
                    plt.ylabel('First Degree')
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()
            
            # 6. Top and bottom influential pairs
            if len(self_influence_scores) >= 2:
                top_n = min(10, len(self_influence_scores) // 2)
                sorted_scores = sorted(self_influence_scores.items(), key=lambda x: x[1], reverse=True)
                top_pairs = sorted_scores[:top_n]
                bottom_pairs = sorted_scores[-top_n:]
                
                plt.figure(figsize=(12, 8))
                
                # Plot top pairs
                plt.subplot(2, 1, 1)
                top_labels = [p[0] for p in reversed(top_pairs)]
                top_values = [p[1] for p in reversed(top_pairs)]
                plt.barh(range(len(top_labels)), top_values)
                plt.yticks(range(len(top_labels)), top_labels)
                plt.title(f'Top {top_n} Most Influential Pairs')
                plt.xlabel('Self-Influence Score')
                
                # Plot bottom pairs
                plt.subplot(2, 1, 2)
                bottom_labels = [p[0] for p in bottom_pairs]
                bottom_values = [p[1] for p in bottom_pairs]
                plt.barh(range(len(bottom_labels)), bottom_values)
                plt.yticks(range(len(bottom_labels)), bottom_labels)
                plt.title(f'Bottom {top_n} Least Influential Pairs')
                plt.xlabel('Self-Influence Score')
                
                plt.tight_layout()
                pdf.savefig()
                plt.close()
            
            # 7. Influence by angle difference
            angle_diff_data = defaultdict(list)
            for pair_id, score in self_influence_scores.items():
                first_deg, second_deg = extract_degrees(pair_id)
                angle_diff = angle_difference(first_deg, second_deg)
                angle_diff_data[angle_diff].append(score)
            
            if angle_diff_data:
                angle_stats = {}
                for angle_diff, scores in angle_diff_data.items():
                    if scores:
                        angle_stats[angle_diff] = {
                            'mean': np.mean(scores),
                            'median': np.median(scores),
                            'std': np.std(scores),
                            'count': len(scores)
                        }
                
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
                    pdf.savefig()
                    plt.close()
    
    # ---- Test Influence Analysis ----
    if test_influence_data:
        # Identify test pairs
        test_pairs = set()
        for train_pair_data in test_influence_data.values():
            for key in train_pair_data.keys():
                if key.startswith('tracin_influence_') and not key.endswith('_per_checkpoint'):
                    test_pairs.add(key.replace('tracin_influence_', ''))
        
        test_pairs = sorted(list(test_pairs))
        
        if test_pairs:
            # 1. Section header for Test Influence
            plt.figure(figsize=(12, 9))
            plt.axis('off')
            plt.text(0.5, 0.5, 'Test Influence Analysis', 
                     horizontalalignment='center', fontsize=24, weight='bold')
            plt.text(0.5, 0.4, f'Total Test Pairs: {len(test_pairs)}', 
                     horizontalalignment='center', fontsize=16)
            pdf.savefig()
            plt.close()
            
            # 2. Analyze influence distribution for test pairs (limit to first 3)
            for test_pair in test_pairs[:min(3, len(test_pairs))]:
                influences = []
                for train_pair_data in test_influence_data.values():
                    key = f'tracin_influence_{test_pair}'
                    if key in train_pair_data:
                        influences.append(train_pair_data[key])
                
                if influences:
                    plt.figure(figsize=(10, 6))
                    sns.histplot(influences, kde=True, bins=min(30, len(set(influences))))
                    
                    # Add vertical lines for statistics
                    plt.axvline(np.mean(influences), color='red', linestyle='--', linewidth=1, 
                                label=f'Mean: {np.mean(influences):.2f}')
                    plt.axvline(np.median(influences), color='green', linestyle='-.', linewidth=1, 
                                label=f'Median: {np.median(influences):.2f}')
                    
                    # Format and label
                    first_deg, second_deg = extract_degrees(test_pair)
                    plt.title(f'Influence Distribution for Test Pair: {first_deg}° → {second_deg}°')
                    plt.xlabel('Influence Score')
                    plt.ylabel('Frequency')
                    plt.legend()
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()
            
            # 3. Most influential training pairs for test pairs
            for test_pair in test_pairs[:min(3, len(test_pairs))]:
                pair_influences = {}
                for train_pair, train_data in test_influence_data.items():
                    key = f'tracin_influence_{test_pair}'
                    if key in train_data:
                        pair_influences[train_pair] = train_data[key]
                
                if pair_influences:
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
                    
                    # Plot bottom pairs
                    plt.subplot(2, 1, 2)
                    bottom_labels = [f"{extract_degrees(p[0])[0]}°→{extract_degrees(p[0])[1]}°" for p in bottom_pairs]
                    bottom_values = [p[1] for p in bottom_pairs]
                    colors = ['green' if v > 0 else 'red' for v in bottom_values]
                    plt.barh(range(len(bottom_labels)), bottom_values, color=colors)
                    plt.yticks(range(len(bottom_labels)), bottom_labels)
                    plt.title(f'Bottom 5 Least Influential Training Pairs on Test Pair {test_first_deg}°→{test_second_deg}°')
                    plt.xlabel('Influence Score')
                    
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()
            
            # 4. Analyze influence by angle difference
            angle_diff_data = defaultdict(list)
            for train_pair, train_data in test_influence_data.items():
                train_first_deg, train_second_deg = extract_degrees(train_pair)
                train_diff = angle_difference(train_first_deg, train_second_deg)
                
                for test_pair in test_pairs:
                    key = f'tracin_influence_{test_pair}'
                    if key in train_data:
                        test_first_deg, test_second_deg = extract_degrees(test_pair)
                        test_diff = angle_difference(test_first_deg, test_second_deg)
                        
                        angle_key = (train_diff, test_diff)
                        angle_diff_data[angle_key].append(train_data[key])
            
            if angle_diff_data:
                angle_stats = {}
                for angle_key, scores in angle_diff_data.items():
                    if scores:
                        angle_stats[angle_key] = {
                            'mean': np.mean(scores),
                            'std': np.std(scores),
                            'count': len(scores)
                        }
                
                if angle_stats:
                    test_angles = sorted(set([k[1] for k in angle_stats.keys()]))
                    
                    plt.figure(figsize=(12, 8))
                    
                    for test_angle in test_angles[:min(4, len(test_angles))]:
                        train_angles = []
                        means = []
                        stds = []
                        
                        for (train_diff, test_diff), stats in angle_stats.items():
                            if test_diff == test_angle:
                                train_angles.append(train_diff)
                                means.append(stats['mean'])
                                stds.append(stats['std'])
                        
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
                    pdf.savefig()
                    plt.close()
    
    # Add a conclusions page
    plt.figure(figsize=(12, 9))
    plt.axis('off')
    plt.text(0.5, 0.9, 'TracIn Analysis Conclusions', 
             horizontalalignment='center', fontsize=24, weight='bold')
    
    # Add key findings
    conclusions = [
        "Self-influence scores reflect sample difficulty and importance during training.",
        "Pairs with higher self-influence scores may require special attention during training.",
        "Angle difference correlates with self-influence, showing the model's sensitivity to angular changes.",
        "Test influence scores show how training samples affect the model's predictions on specific test pairs.",
        "Both positive and negative influence scores were observed, indicating that some training samples can hurt performance on specific test samples.",
        "These insights can guide dataset curation, augmentation strategies, and model improvements."
    ]
    
    y_pos = 0.8
    for conclusion in conclusions:
        plt.text(0.1, y_pos, "• " + conclusion, fontsize=14, wrap=True)
        y_pos -= 0.1
    
    # Add recommendations
    plt.text(0.5, 0.3, 'Recommendations', 
             horizontalalignment='center', fontsize=20, weight='bold')
    
    recommendations = [
        "Focus data collection on underrepresented or difficult angle pairs.",
        "Consider curriculum learning approaches using the influence difficulty order.",
        "Investigate and potentially remove harmful training samples with negative influence.",
        "Use these insights to guide model architecture decisions."
    ]
    
    y_pos = 0.25
    for recommendation in recommendations:
        plt.text(0.1, y_pos, "• " + recommendation, fontsize=14, wrap=True)
        y_pos -= 0.05
    
    pdf.savefig()
    plt.close()

print(f"PDF report generated: {report_filename}") 