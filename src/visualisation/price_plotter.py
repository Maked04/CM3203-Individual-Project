import sys
import os
from src.data.loader import load_token_price_data, get_token_stats, list_available_tokens
import matplotlib.pyplot as plt
import random
import math
import numpy as np


def plot_token_price(token_addresses: str | list, title=None):
    """
    Create price plots for one or more tokens.
   
    Args:
        token_addresses: Single token address (str) or list of token addresses
        title: Optional title for the overall figure
    """
    # Convert single token to list for consistent handling
    if isinstance(token_addresses, str):
        token_addresses = [token_addresses]
   
    num_tokens = len(token_addresses)
    if num_tokens == 0:
        print("No tokens provided")
        return
   
    # Calculate subplot grid dimensions
    if num_tokens == 1:
        rows, cols = 1, 1
    else:
        cols = min(2, num_tokens)  # Maximum 2 columns
        rows = math.ceil(num_tokens / cols)
   
    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(12*cols, 6*rows))
   
    # Convert axes to array if single plot
    if num_tokens == 1:
        axes = np.array([axes])
   
    # Flatten axes array for easier iteration
    axes_flat = axes.flatten() if num_tokens > 1 else [axes]
   
    # Plot each token
    for idx, (ax, token_address) in enumerate(zip(axes_flat, token_addresses)):
        # Load the data
        price_data = load_token_price_data(token_address)
       
        if price_data is None:
            ax.text(0.5, 0.5, f"No data available for\n{token_address}",
                   ha='center', va='center')
            continue
       
        # Get statistics
        stats = get_token_stats(price_data)
       
        # Plot price data
        ax.plot(price_data.index, price_data['price'], linewidth=2)
       
        # Add labels and title
        ax.set_title(f'Token Price Over Time\n{token_address}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price (SOL)')
       
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
       
        # Rotate x-axis labels
        ax.tick_params(axis='x', rotation=45)
       
        # Add stats annotation
        stats_text = (
            f"Initial Price: {stats['initial_price']:.8f}\n"
            f"Final Price: {stats['final_price']:.8f}\n"
            f"Change: {stats['price_change_pct']:.2f}%"
        )
        ax.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                   bbox=dict(facecolor='white', alpha=0.8),
                   verticalalignment='top')
   
    # Hide empty subplots if any
    for idx in range(len(token_addresses), len(axes_flat)):
        axes_flat[idx].axis('off')
   
    # Add main title to the figure if provided
    if title:
        fig.suptitle(title, fontsize=16, y=1.02)
   
    # Adjust layout
    plt.tight_layout()
   
    # Show plot
    plt.show()


def compare_token_price_anomalies(token_address: str):
    """
    Create a side-by-side comparison of token price with and without anomaly removal.
    
    Args:
        token_address (str): Token address to analyze
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6))
    
    # Plot with anomaly removal
    price_data_cleaned = load_token_price_data(token_address, remove_anomalies=True)
    if price_data_cleaned is not None:
        stats_cleaned = get_token_stats(price_data_cleaned)
        
        ax1.plot(price_data_cleaned.index, price_data_cleaned['price'], linewidth=2)
        ax1.set_title(f'Price With Anomaly Removal\n{token_address}')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price (SOL)')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.tick_params(axis='x', rotation=45)
        
        stats_text = (
            f"Initial Price: {stats_cleaned['initial_price']:.8f}\n"
            f"Final Price: {stats_cleaned['final_price']:.8f}\n"
            f"Change: {stats_cleaned['price_change_pct']:.2f}%"
        )
        ax1.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                    bbox=dict(facecolor='white', alpha=0.8),
                    verticalalignment='top')
    
    # Plot without anomaly removal
    price_data_raw = load_token_price_data(token_address, remove_anomalies=False)
    if price_data_raw is not None:
        stats_raw = get_token_stats(price_data_raw)
        
        ax2.plot(price_data_raw.index, price_data_raw['price'], linewidth=2)
        ax2.set_title(f'Price Without Anomaly Removal\n{token_address}')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Price (SOL)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.tick_params(axis='x', rotation=45)
        
        stats_text = (
            f"Initial Price: {stats_raw['initial_price']:.8f}\n"
            f"Final Price: {stats_raw['final_price']:.8f}\n"
            f"Change: {stats_raw['price_change_pct']:.2f}%"
        )
        ax2.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                    bbox=dict(facecolor='white', alpha=0.8),
                    verticalalignment='top')
    
    # Add comparison statistics if both plots are available
    if price_data_cleaned is not None and price_data_raw is not None:
        removed_points = len(price_data_raw) - len(price_data_cleaned)
        percent_removed = (removed_points / len(price_data_raw)) * 100
        
        comparison_text = (
            f"Anomaly Removal Stats:\n"
            f"Points Removed: {removed_points}\n"
            f"Percent Removed: {percent_removed:.1f}%"
        )
        fig.text(0.5, 0.02, comparison_text, ha='center', 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":    # Example usage
    tokens = list_available_tokens()
    token_address = random.choice(tokens)
    print(f"Comparing anomaly removal for token: {token_address}")
    compare_token_price_anomalies(token_address)