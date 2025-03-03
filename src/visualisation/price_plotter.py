import sys
import os
from src.data.loader import load_token_price_data, get_token_stats, list_available_tokens
from src.data.processor import trim_main_trading_period
import matplotlib.pyplot as plt
import random
import math
import numpy as np
import pandas as pd


import numpy as np
import math
import matplotlib.pyplot as plt

def plot_token_price(token_addresses: str | list, title=None, trim_dead_period=False):
    """
    Create price plots for one or more tokens.
   
    Args:
        token_addresses: Single token address (str) or list of token addresses
        title: Optional title for the overall figure
    """
    if isinstance(token_addresses, str):
        token_addresses = [token_addresses]
   
    num_tokens = len(token_addresses)
    if num_tokens == 0:
        print("No tokens provided")
        return
   
    cols = min(2, num_tokens)  # Max 2 columns
    rows = math.ceil(num_tokens / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(12 * cols, 6 * rows))

    # Ensure axes is always iterable
    if num_tokens == 1:
        axes = [axes]  # Convert single AxesSubplot to a list
    else:
        axes = axes.flatten()  # Flatten for consistent iteration
   
    for idx, (ax, token_address) in enumerate(zip(axes, token_addresses)):
        price_data = load_token_price_data(token_address)

        if trim_dead_period:
            price_data = trim_main_trading_period(price_data, min_trades_per_hour=100, window_size_hours=0.5)
       
        if price_data is None:
            ax.text(0.5, 0.5, f"No data available for\n{token_address}",
                    ha='center', va='center')
            continue
       
        stats = get_token_stats(price_data)
        ax.plot(price_data.index, price_data['price'], linewidth=2)
        ax.set_title(f'Token Price Over Time\n{token_address}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price (SOL)')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='x', rotation=45)

        stats_text = (
            f"Initial Price: {stats['initial_price']:.8f}\n"
            f"Final Price: {stats['final_price']:.8f}\n"
            f"Change: {stats['price_change_pct']:.2f}%"
        )
        ax.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                    bbox=dict(facecolor='white', alpha=0.8),
                    verticalalignment='top')
   
    for idx in range(len(token_addresses), len(axes)):
        axes[idx].axis('off')
   
    if title:
        fig.suptitle(title, fontsize=16, y=1.02)

    plt.tight_layout()
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


def plot_token_before_after_trim(original_df, trimmed_df, token_address=None):
    """
    Plot token data before and after applying the trim_dead_time function.
    
    Args:
        original_df: Original DataFrame with price data
        trimmed_df: DataFrame after applying trim_dead_time
        token_address: Optional token address for labeling
    """
    # Set up the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Get token address from dataframe attrs if available and not provided
    if token_address is None and hasattr(original_df, 'attrs') and 'token_address' in original_df.attrs:
        token_address = original_df.attrs['token_address']
    elif token_address is None:
        token_address = "Unknown Token"
    
    # Make sure data is sorted by timestamp (oldest to newest)
    original_df = original_df.sort_index()
    trimmed_df = trimmed_df.sort_index()
    
    # Plot original data
    ax1.plot(original_df.index, original_df['token_price'], linewidth=2, color='blue')
    ax1.set_title(f'Token Price Before Trimming\n{token_address}')
    ax1.set_xlabel('Time (Unix Timestamp)')
    ax1.set_ylabel('Price')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Set x-axis limits explicitly for original data
    x_min_orig = original_df.index.min()
    x_max_orig = original_df.index.max()
    ax1.set_xlim(x_min_orig, x_max_orig)
    
    # Add annotation for original data
    original_stats = {
        'data_points': len(original_df),
        'time_range': f"{x_min_orig} - {x_max_orig}",
        'duration': f"{(x_max_orig - x_min_orig) / 86400:.2f} days"
    }
    
    stats_text1 = (
        f"Data Points: {original_stats['data_points']}\n"
        f"Time Range: {original_stats['time_range']}\n"
        f"Duration: {original_stats['duration']}"
    )
    ax1.annotate(stats_text1, xy=(0.02, 0.98), xycoords='axes fraction',
               bbox=dict(facecolor='white', alpha=0.8),
               verticalalignment='top')
    
    # Plot trimmed data
    ax2.plot(trimmed_df.index, trimmed_df['token_price'], linewidth=2, color='green')
    ax2.set_title(f'Token Price After Trimming\n{token_address}')
    ax2.set_xlabel('Time (Unix Timestamp)')
    ax2.set_ylabel('Price')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Set x-axis limits explicitly for trimmed data
    x_min_trim = trimmed_df.index.min()
    x_max_trim = trimmed_df.index.max()
    ax2.set_xlim(x_min_trim, x_max_trim)
    
    # Add annotation for trimmed data
    trimmed_stats = {
        'data_points': len(trimmed_df),
        'time_range': f"{x_min_trim} - {x_max_trim}",
        'duration': f"{(x_max_trim - x_min_trim) / 86400:.2f} days",
        'removed_points': len(original_df) - len(trimmed_df)
    }
    
    stats_text2 = (
        f"Data Points: {trimmed_stats['data_points']}\n"
        f"Time Range: {trimmed_stats['time_range']}\n"
        f"Duration: {trimmed_stats['duration']}\n"
        f"Removed Points: {trimmed_stats['removed_points']} ({trimmed_stats['removed_points']/len(original_df)*100:.1f}%)"
    )
    ax2.annotate(stats_text2, xy=(0.02, 0.98), xycoords='axes fraction',
               bbox=dict(facecolor='white', alpha=0.8),
               verticalalignment='top')
    
    # For better readability, convert timestamps to readable dates in x-tick labels
    def format_timestamp(x, pos):
        return pd.to_datetime(x, unit='s').strftime('%Y-%m-%d')
    
    from matplotlib.ticker import FuncFormatter
    formatter = FuncFormatter(format_timestamp)
    ax1.xaxis.set_major_formatter(formatter)
    ax2.xaxis.set_major_formatter(formatter)
    
    # Adjust layout
    plt.tight_layout()
    
    # Show plot
    plt.show()
    
    return fig


def plot_token_lifecycle(full_df, launch_df, peak_df, decline_df, token_address=None):
    """
    Plot token data with lifecycle phases highlighted in different colors.
    
    Args:
        full_df: Complete DataFrame with price data
        launch_df: Launch phase DataFrame
        peak_df: Peak phase DataFrame
        decline_df: Decline phase DataFrame
        token_address: Optional token address for labeling
    """
    # Set up the figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get token address from dataframe attrs if available and not provided
    if token_address is None and hasattr(full_df, 'attrs') and 'token_address' in full_df.attrs:
        token_address = full_df.attrs['token_address']
    elif token_address is None:
        token_address = "Unknown Token"
    
    # Make sure data is sorted by timestamp (oldest to newest)
    full_df = full_df.sort_index()
    
    # Plot full data as a light gray background line
    ax.plot(full_df.index, full_df['token_price'], linewidth=1.5, color='lightgray', 
            alpha=0.5, label='All Data')
    
    # Plot each phase with different colors
    if not launch_df.empty:
        ax.plot(launch_df.index, launch_df['token_price'], linewidth=2.5, color='green', 
                label='Launch Phase')
    
    if not peak_df.empty:
        ax.plot(peak_df.index, peak_df['token_price'], linewidth=2.5, color='red', 
                label='Peak Phase')
    
    if not decline_df.empty:
        ax.plot(decline_df.index, decline_df['token_price'], linewidth=2.5, color='blue', 
                label='Decline Phase')
    
    # Add title and labels
    ax.set_title(f'Token Lifecycle Phases\n{token_address}', fontsize=14)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Price (SOL)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set x-axis limits explicitly
    x_min = full_df.index.min()
    x_max = full_df.index.max()
    ax.set_xlim(x_min, x_max)
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Add phase statistics
    stats = {
        'full': {
            'data_points': len(full_df),
            'duration': f"{(full_df.index.max() - full_df.index.min()) / 86400:.2f} days",
            'max_price': f"{full_df['token_price'].max():.8f}",
        },
        'launch': {
            'data_points': len(launch_df),
            'duration': f"{(launch_df.index.max() - launch_df.index.min()) / 86400:.2f} days" if len(launch_df) > 0 else "N/A",
        },
        'peak': {
            'data_points': len(peak_df),
            'duration': f"{(peak_df.index.max() - peak_df.index.min()) / 86400:.2f} days" if len(peak_df) > 0 else "N/A",
        },
        'decline': {
            'data_points': len(decline_df),
            'duration': f"{(decline_df.index.max() - decline_df.index.min()) / 86400:.2f} days" if len(decline_df) > 0 else "N/A",
        }
    }
    
    stats_text = (
        f"Total Data Points: {stats['full']['data_points']}\n"
        f"Total Duration: {stats['full']['duration']}\n"
        f"Max Price: {stats['full']['max_price']}\n\n"
        f"Launch Phase: {stats['launch']['data_points']} points, {stats['launch']['duration']}\n"
        f"Peak Phase: {stats['peak']['data_points']} points, {stats['peak']['duration']}\n"
        f"Decline Phase: {stats['decline']['data_points']} points, {stats['decline']['duration']}"
    )
    
    # Add text box with statistics
    ax.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
               bbox=dict(facecolor='white', alpha=0.8),
               verticalalignment='top')
    
    # For better readability, convert timestamps to readable dates in x-tick labels
    def format_timestamp(x, pos):
        return pd.to_datetime(x, unit='s').strftime('%Y-%m-%d')
    
    from matplotlib.ticker import FuncFormatter
    formatter = FuncFormatter(format_timestamp)
    ax.xaxis.set_major_formatter(formatter)
    
    # Add markers for phase transitions
    if not launch_df.empty:
        launch_end = launch_df.index.max()
        ax.axvline(x=launch_end, color='green', linestyle='--', alpha=0.5)
    
    if not peak_df.empty:
        peak_start = peak_df.index.min()
        peak_end = peak_df.index.max()
        
        # Only add these lines if they don't overlap with launch end
        if not launch_df.empty and peak_start > launch_df.index.max():
            ax.axvline(x=peak_start, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=peak_end, color='red', linestyle='--', alpha=0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Show plot
    plt.show()
    
    return fig

if __name__ == "__main__":    # Example usage
    tokens = list_available_tokens()
    token_address = random.choice(tokens)
    print(f"Comparing anomaly removal for token: {token_address}")
    compare_token_price_anomalies(token_address)