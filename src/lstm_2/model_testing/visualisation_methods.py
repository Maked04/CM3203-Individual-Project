import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_error_distribution(pred, real, bins=30, figsize=(14, 8), title='Error Distribution'):
    # Convert inputs to numpy arrays if they aren't already
    pred = np.array(pred)
    real = np.array(real)
    
    # Calculate errors (residuals)
    errors = real - pred
   
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
   
    # Plot 1: Error distribution
    ax1.hist(errors, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=1)
    
    # Add mean error line
    mean_error = np.mean(errors)
    ax1.axvline(x=mean_error, color='green', linestyle='-', linewidth=1.5,
              label=f'Mean Error: {mean_error:.4f}')
   
    # Add labels and title for error distribution
    ax1.set_xlabel('Error (Actual - Predicted)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Error Distribution')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
   
    # Add summary statistics as text for error distribution
    stats_text = f'Mean Error: {np.mean(errors):.4f}\n'
    stats_text += f'Std Dev: {np.std(errors):.4f}\n'
    stats_text += f'Median Error: {np.median(errors):.4f}\n'
    stats_text += f'Min Error: {np.min(errors):.4f}\n'
    stats_text += f'Max Error: {np.max(errors):.4f}'
   
    # Position text in the upper right of first subplot
    ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Real values distribution
    ax2.hist(pred, bins=bins, alpha=0.7, color='lightgreen', edgecolor='black')
    
    # Add mean line for real values
    mean_pred = np.mean(pred)
    ax2.axvline(x=mean_pred, color='blue', linestyle='-', linewidth=1.5,
               label=f'Mean: {mean_pred:.4f}')
    
    # Add labels and title for pred values distribution
    ax2.set_xlabel('Pred Values')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Predicted Values')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add summary statistics as text for real values
    real_stats_text = f'Mean: {np.mean(real):.4f}\n'
    real_stats_text += f'Std Dev: {np.std(real):.4f}\n'
    real_stats_text += f'Median: {np.median(real):.4f}\n'
    real_stats_text += f'Min: {np.min(real):.4f}\n'
    real_stats_text += f'Max: {np.max(real):.4f}'
    
    # Position text in the upper right of second subplot
    ax2.text(0.95, 0.95, real_stats_text, transform=ax2.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
   
    # Add overall title
    fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust to make room for the suptitle
    plt.show()


def plot_directional_accuracy(pred, real, bins=None):
    if bins is None:
        bins = [-np.inf, -1, -0.5, -0.1, 0, 0.1, 0.5, 1, np.inf]  # Default bins including sign
    pred = np.array(pred).flatten()
    real = np.array(real).flatten()
   
    bin_labels = [f"{bins[i]} to {bins[i+1]}" for i in range(len(bins)-1)]
    # Assign Predicted Values to Bins
    binned_data = pd.cut(pred, bins, labels=bin_labels)
    
    # Compute Directional Accuracy per Bin
    bin_accuracy = {}
    bin_counts = {}
    for bin_label in bin_labels:
        indices = (binned_data == bin_label)
        bin_counts[bin_label] = indices.sum()
        
        if indices.sum() > 0:  # Avoid division by zero
            correct_signs = np.sum(np.sign(pred[indices]) == np.sign(real[indices]))
            accuracy = correct_signs / indices.sum()
            bin_accuracy[bin_label] = accuracy
        else:
            bin_accuracy[bin_label] = None  # No data points in this bin
    
    # Remove empty bins
    bin_accuracy = {k: v for k, v in bin_accuracy.items() if v is not None}
    bin_counts = {k: v for k, v in bin_counts.items() if v > 0}
    
    # Prepare data for plotting
    labels = list(bin_accuracy.keys())
    accuracies = list(bin_accuracy.values())
    counts = [bin_counts[label] for label in labels]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, accuracies, color='skyblue')
    plt.xlabel("Predicted Value Bins")
    plt.ylabel("Directional Accuracy")
    plt.title("Directional Accuracy per Bin (with Prediction Counts)")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    
    # Add count labels on top of each bar
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'n={count}',
                 ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    return bin_accuracy, bin_counts