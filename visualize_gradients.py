import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("talk")

# Define paths
base_path = "experiments/svrg_comparison/results/svrg_comparison_20250417_182014"
methods = ["standard", "svrg"]
frequencies = ["500hz", "1000hz", "3000hz"]

# Create output directory
os.makedirs("gradient_visualizations", exist_ok=True)

# Function to load gradient data
def load_gradient_data(method, freq):
    file_path = os.path.join(base_path, method, freq, "gradients", "gradient_stats.csv")
    print(f"Attempting to load: {file_path}")
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            print(f"Successfully loaded {method} {freq} data: {len(df)} rows")
            # Print column names to debug
            print(f"Columns: {df.columns.tolist()}")
            return df
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    else:
        print(f"Data file not found: {file_path}")
        return None

# Check what files actually exist in the directories
def check_file_structure():
    print("\nChecking file structure...")
    for method in methods:
        for freq in frequencies:
            grad_dir = os.path.join(base_path, method, freq, "gradients")
            if os.path.exists(grad_dir):
                files = os.listdir(grad_dir)
                print(f"{method}/{freq}/gradients: {len(files)} files")
                # Print first few files
                for f in files[:3]:
                    print(f"  - {f}")
            else:
                print(f"{method}/{freq}/gradients: Directory not found")

# Set up figure for gradient norm comparison
def plot_gradient_comparison():
    print("\nGenerating gradient comparison plots...")
    fig, axes = plt.subplots(len(frequencies), 1, figsize=(12, 5*len(frequencies)), sharex=True)
    
    if len(frequencies) == 1:
        axes = [axes]
        
    for i, freq in enumerate(frequencies):
        ax = axes[i]
        
        # Load data for both methods
        standard_data = load_gradient_data("standard", freq)
        svrg_data = load_gradient_data("svrg", freq)
        
        if standard_data is not None and svrg_data is not None:
            # Plot gradient norms
            ax.plot(standard_data['step_count'], standard_data['grad_norm'], 
                   label='Standard Grad Norm', color='blue', alpha=0.7)
            ax.plot(svrg_data['step_count'], svrg_data['grad_norm'], 
                   label='SVRG Grad Norm', color='red', alpha=0.7)
            
            # Add correction norm for SVRG
            if 'correction_norm' in svrg_data.columns:
                ax.plot(svrg_data['step_count'], svrg_data['correction_norm'], 
                       label='SVRG Correction Norm', color='green', alpha=0.5, linestyle='--')
            
            # Add rolling average for smoother visualization
            window = 5
            if len(standard_data) > window:
                ax.plot(standard_data['step_count'], 
                       standard_data['grad_norm'].rolling(window=window).mean(), 
                       color='darkblue', linewidth=2, label='Standard Rolling Avg')
            
            if len(svrg_data) > window:
                ax.plot(svrg_data['step_count'], 
                       svrg_data['grad_norm'].rolling(window=window).mean(), 
                       color='darkred', linewidth=2, label='SVRG Rolling Avg')
        else:
            ax.text(0.5, 0.5, f"Data not available for {freq}", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
        
        ax.set_title(f'Gradient Norm Comparison - {freq}')
        ax.set_ylabel('Norm Value')
        ax.legend(loc='upper right')
        
    plt.xlabel('Step Count')
    plt.tight_layout()
    
    save_path = 'gradient_visualizations/gradient_norm_comparison.png'
    print(f"Saving plot to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close()

# Plot gradient stats over training
def plot_gradient_stats_by_method():
    print("\nGenerating gradient stats plots by method...")
    for method in methods:
        fig, axes = plt.subplots(len(frequencies), 1, figsize=(12, 5*len(frequencies)), sharex=True)
        
        if len(frequencies) == 1:
            axes = [axes]
            
        for i, freq in enumerate(frequencies):
            ax = axes[i]
            data = load_gradient_data(method, freq)
            
            if data is not None:
                # Standard method columns
                if method == 'standard':
                    # Plot mean and std of gradient norm
                    if 'avg_grad_norm' in data.columns:
                        ax.plot(data['step_count'], data['avg_grad_norm'], 
                              label='Avg Grad Norm', color='blue')
                        ax.fill_between(data['step_count'], 
                                       data['avg_grad_norm'] - data['std_grad_norm'],
                                       data['avg_grad_norm'] + data['std_grad_norm'],
                                       alpha=0.3, color='blue')
                    
                    # Plot max gradient
                    if 'max_grad_norm' in data.columns:
                        ax.plot(data['step_count'], data['max_grad_norm'], 
                              label='Max Grad Norm', color='red', linestyle=':')
                
                # SVRG method columns 
                elif method == 'svrg':
                    # For SVRG, plot grad norm
                    ax.plot(data['step_count'], data['grad_norm'], 
                           label='Grad Norm', color='blue')
                    
                    # Plot correction norms
                    if 'correction_norm' in data.columns:
                        ax.plot(data['step_count'], data['correction_norm'], 
                               label='Correction Norm', color='green')
                        
                        # Plot average correction norm if available
                        if 'avg_correction_norm' in data.columns:
                            ax.plot(data['step_count'], data['avg_correction_norm'], 
                                   label='Avg Correction Norm', color='orange', linestyle='--')
                            
                            # Add std deviation as shaded area if available
                            if 'std_correction_norm' in data.columns:
                                ax.fill_between(data['step_count'], 
                                               data['avg_correction_norm'] - data['std_correction_norm'],
                                               data['avg_correction_norm'] + data['std_correction_norm'],
                                               alpha=0.2, color='orange')
            else:
                ax.text(0.5, 0.5, f"Data not available for {method} {freq}", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes)
            
            ax.set_title(f'{method.capitalize()} Method - {freq}')
            ax.set_ylabel('Norm Value')
            ax.legend(loc='upper right')
            
        plt.xlabel('Step Count')
        plt.tight_layout()
        
        save_path = f'gradient_visualizations/{method}_gradient_stats.png'
        print(f"Saving plot to {save_path}")
        plt.savefig(save_path, dpi=300)
        plt.close()

# Plot epoch-wise gradient comparison
def plot_epoch_comparison():
    print("\nGenerating epoch comparison plots...")
    # Extract epoch-wise data
    for freq in frequencies:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        epoch_data = {}
        data_available = False
        
        for method in methods:
            data = load_gradient_data(method, freq)
            if data is not None:
                # Group by epoch and calculate mean gradient norm
                epoch_means = data.groupby('epoch')['grad_norm'].mean()
                epoch_data[method] = epoch_means
                data_available = True
        
        if data_available:
            # Plot epoch means for both methods
            for method, means in epoch_data.items():
                ax.plot(means.index, means.values, 
                       label=f'{method.capitalize()} Method', 
                       marker='o' if method == 'standard' else 's')
            
            ax.set_title(f'Epoch-wise Average Gradient Norm - {freq}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Average Gradient Norm')
            ax.legend()
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, f"No data available for {freq}", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
        
        plt.tight_layout()
        
        save_path = f'gradient_visualizations/epoch_comparison_{freq}.png'
        print(f"Saving plot to {save_path}")
        plt.savefig(save_path, dpi=300)
        plt.close()

# Create a heatmap of gradients across epochs and batches
def plot_gradient_heatmap():
    print("\nGenerating gradient heatmap plots...")
    for method in methods:
        for freq in frequencies:
            data = load_gradient_data(method, freq)
            
            if data is not None:
                try:
                    # Pivot data to create a matrix of epochs and batches
                    pivot_data = data.pivot_table(index='epoch', columns='batch_idx', values='grad_norm')
                    
                    plt.figure(figsize=(12, 8))
                    sns.heatmap(pivot_data, cmap='viridis', annot=False)
                    plt.title(f'{method.capitalize()} Method - {freq} - Gradient Norm Heatmap')
                    plt.ylabel('Epoch')
                    plt.xlabel('Batch Index')
                    
                    save_path = f'gradient_visualizations/{method}_{freq}_heatmap.png'
                    print(f"Saving heatmap to {save_path}")
                    plt.tight_layout()
                    plt.savefig(save_path, dpi=300)
                    plt.close()
                except Exception as e:
                    print(f"Error creating heatmap for {method} {freq}: {e}")

# Additional plot: Ratio of SVRG correction to gradient norm
def plot_correction_ratio():
    print("\nGenerating SVRG correction ratio plots...")
    fig, axes = plt.subplots(len(frequencies), 1, figsize=(12, 5*len(frequencies)), sharex=True)
    
    if len(frequencies) == 1:
        axes = [axes]
        
    for i, freq in enumerate(frequencies):
        ax = axes[i]
        data = load_gradient_data("svrg", freq)
        
        if data is not None and 'correction_norm' in data.columns and 'grad_norm' in data.columns:
            # Calculate ratio of correction to gradient norm
            data['correction_ratio'] = data['correction_norm'] / data['grad_norm']
            
            # Plot ratio
            ax.plot(data['step_count'], data['correction_ratio'], color='purple')
            
            # Add rolling average
            window = 5
            if len(data) > window:
                ax.plot(data['step_count'], 
                       data['correction_ratio'].rolling(window=window).mean(), 
                       color='darkviolet', linewidth=2, label='Rolling Average')
            
            # Add horizontal line at ratio=1
            ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
            
            ax.set_title(f'SVRG Correction/Gradient Ratio - {freq}')
            ax.set_ylabel('Ratio')
            ax.legend(loc='upper right')
        else:
            ax.text(0.5, 0.5, f"Data not available for {freq}", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            
    plt.xlabel('Step Count')
    plt.tight_layout()
    save_path = 'gradient_visualizations/svrg_correction_ratio.png'
    print(f"Saving plot to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close()

# Run all visualizations
if __name__ == "__main__":
    print("Starting gradient visualization process...")
    print(f"Base path: {base_path}")
    print(f"Methods: {methods}")
    print(f"Frequencies: {frequencies}")
    
    # First check what files actually exist
    check_file_structure()
    
    # Now run the visualizations
    plot_gradient_comparison()
    plot_gradient_stats_by_method()
    plot_epoch_comparison()
    plot_gradient_heatmap()
    plot_correction_ratio()  # New plot for SVRG
    
    print("\nVisualization process completed. Results in 'gradient_visualizations/' folder.") 