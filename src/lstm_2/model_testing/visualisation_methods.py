import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.data_processing.loader import load_token_price_data
from src.lstm_2.model_testing.metrics_methods import large_move_metrics
import plotly.graph_objects as go
from matplotlib.gridspec import GridSpec
import seaborn as sns


def plot_error_distribution(pred, real, bins=30, title='Error Distribution'):
    pred = np.array(pred)
    real = np.array(real)
    
    errors = real - pred
    mean_error = np.mean(errors)

    plt.figure(figsize=(10, 5))
    plt.hist(errors, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1, label='Zero Error')
    plt.axvline(x=mean_error, color='green', linestyle='-', linewidth=1.5, label=f'Mean Error: {mean_error:.4f}')

    plt.xlabel('Error (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add error stats box
    error_stats = f'Mean: {mean_error:.4f}\n' \
                  f'Std Dev: {np.std(errors):.4f}\n' \
                  f'Median: {np.median(errors):.4f}\n' \
                  f'Min: {np.min(errors):.4f}\n' \
                  f'Max: {np.max(errors):.4f}'
    plt.text(0.95, 0.95, error_stats, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

def plot_distribution(values, bins=30, name="", color='lightgreen'):
    values = np.array(values)

    plt.figure(figsize=(10, 5))
    plt.hist(values, bins=bins, alpha=0.7, color=color, edgecolor='black')
    mean_pred = np.mean(values)
    plt.axvline(x=mean_pred, color='black', linestyle='-', linewidth=1.5, label=f'Mean: {mean_pred:.4f}')
    plt.xlabel(f'{name}Values')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {name} Values')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add stats for predictions
    pred_stats = f'Mean: {mean_pred:.4f}\n' \
                 f'Std Dev: {np.std(values):.4f}\n' \
                 f'Median: {np.median(values):.4f}\n' \
                 f'Min: {np.min(values):.4f}\n' \
                 f'Max: {np.max(values):.4f}'
    plt.text(0.95, 0.95, pred_stats, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

def plot_value_distributions(pred, real, bins=30):
    pred = np.array(pred)
    real = np.array(real)

    # -------- Plot 1: Distribution of Predicted Values --------
    plt.figure(figsize=(10, 5))
    plt.hist(pred, bins=bins, alpha=0.7, color='lightgreen', edgecolor='black')
    mean_pred = np.mean(pred)
    plt.axvline(x=mean_pred, color='blue', linestyle='-', linewidth=1.5, label=f'Mean: {mean_pred:.4f}')
    plt.xlabel('Predicted Values')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Values')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add stats for predictions
    pred_stats = f'Mean: {mean_pred:.4f}\n' \
                 f'Std Dev: {np.std(pred):.4f}\n' \
                 f'Median: {np.median(pred):.4f}\n' \
                 f'Min: {np.min(pred):.4f}\n' \
                 f'Max: {np.max(pred):.4f}'
    plt.text(0.95, 0.95, pred_stats, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

    # -------- Plot 2: Distribution of Real Values --------
    plt.figure(figsize=(10, 5))
    plt.hist(real, bins=bins, alpha=0.7, color='orange', edgecolor='black')
    mean_real = np.mean(real)
    plt.axvline(x=mean_real, color='red', linestyle='-', linewidth=1.5, label=f'Mean: {mean_real:.4f}')
    plt.xlabel('Real Values')
    plt.ylabel('Frequency')
    plt.title('Distribution of Real Values')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add stats for actuals
    real_stats = f'Mean: {mean_real:.4f}\n' \
                 f'Std Dev: {np.std(real):.4f}\n' \
                 f'Median: {np.median(real):.4f}\n' \
                 f'Min: {np.min(real):.4f}\n' \
                 f'Max: {np.max(real):.4f}'
    plt.text(0.95, 0.95, real_stats, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
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


def plot_magnitude_accuracy(pred, real, bins=None, use_percentage_error=False):
    if bins is None:
        bins = [-np.inf, -1, -0.5, -0.1, 0.1, 0.5, 1, np.inf]  # Exclude zero to avoid division issues

    pred = np.array(pred).flatten()
    real = np.array(real).flatten()
    
    # Filter out values where prediction is exactly 0 if using percentage error
    if use_percentage_error:
        nonzero_mask = pred != 0
        pred = pred[nonzero_mask]
        real = real[nonzero_mask]

    bin_labels = [f"{bins[i]} to {bins[i+1]}" for i in range(len(bins)-1)]
    binned_data = pd.cut(pred, bins, labels=bin_labels)

    bin_errors = {}
    bin_counts = {}

    for bin_label in bin_labels:
        indices = (binned_data == bin_label)
        bin_counts[bin_label] = indices.sum()

        if indices.sum() > 0:
            if use_percentage_error:
                errors = np.abs((real[indices] - pred[indices]) / pred[indices])
            else:
                errors = np.abs(real[indices] - pred[indices])
            avg_error = np.mean(errors)
            bin_errors[bin_label] = avg_error
        else:
            bin_errors[bin_label] = None

    # Remove empty bins
    bin_errors = {k: v for k, v in bin_errors.items() if v is not None}
    bin_counts = {k: v for k, v in bin_counts.items() if v > 0}

    labels = list(bin_errors.keys())
    avg_errors = list(bin_errors.values())
    counts = [bin_counts[label] for label in labels]

    # Plotting
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, avg_errors, color='salmon')
    plt.xlabel("Predicted Value Bins")
    ylabel = "Average Percentage Error" if use_percentage_error else "Average Absolute Error"
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} per Bin (with Prediction Counts)")
    plt.xticks(rotation=45, ha='right')

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'n={count}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

    return bin_errors, bin_counts


def plot_prediction_vs_actual(pred, real):
    plt.figure(figsize=(10, 6))
    plt.scatter(real, pred, alpha=0.5)
    plt.plot([real.min(), real.max()], [real.min(), real.max()], 'r--', lw=2)
    plt.title('Predicted vs Actual Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.tight_layout()
    plt.show()


def plot_predictions_on_price_graph(token_address, bucket_pred_map, min_abs_pred_size=0.5):

    # Load and process price data
    price_df = load_token_price_data(token_address, use_datetime=False)
    if price_df is None or price_df.empty:
        print("No price data available to plot.")
        return

    # Aggregate duplicate timestamps by averaging
    price_df = price_df.groupby(price_df.index).mean()

    # Base price line
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_df.index,
        y=price_df['price'],
        mode='lines',
        name='Token Price',
        line=dict(color='black')
    ))

    # Add predictions as arrows/markers
    for pred_data in bucket_pred_map:
        # Extract bucket time and prediction from the list of dicts
        bucket_time = pred_data['bucket_time']
        pred = pred_data['prediction']

        if abs(pred) < min_abs_pred_size:
            continue

        start_time, end_time = bucket_time

        # Handle missing times
        if end_time not in price_df.index:
            closest_idx = price_df.index.get_indexer([end_time], method='nearest')[0]
            closest_time = price_df.index[closest_idx]
        else:
            closest_time = end_time

        price_at_end = price_df.loc[closest_time, 'price']
        color = 'green' if pred > 0 else 'red'

        fig.add_trace(go.Scatter(
            x=[closest_time],
            y=[price_at_end],
            mode='markers+text',
            marker=dict(color=color, size=10, symbol='arrow-up' if pred > 0 else 'arrow-down'),
            text=[f'{pred:.2f}x'],
            textposition='top center',
            name='Prediction'
        ))

    # Layout tweaks
    fig.update_layout(
        title=f"Token Price with Predictions: {token_address}",
        xaxis_title="Timestamp",
        yaxis_title="Price",
        hovermode="x unified",
        template="plotly_white"
    )

    fig.show()


def plot_model_comparison(model_comparison, sort_by='combined_metric', include_metrics=None):
    """
    Create visualizations to compare multiple models in two sections:
    
    Section 1:
    - Model ranking by combined score
    - Performance metrics heatmap
    
    Section 2:
    - Detailed metrics bar chart
    - Error metrics comparison
   
    Parameters:
    - model_comparison: list of dictionaries or DataFrame with model metrics
    - sort_by: metric to sort models by (default: combined_metric)
    - include_metrics: list of specific metrics to include (defaults to most important if None)
   
    Returns:
    - None (displays plots)
    """
    # Convert to DataFrame if list of dictionaries
    if isinstance(model_comparison, list):
        df_comparison = pd.DataFrame(model_comparison)
    else:
        df_comparison = model_comparison.copy()
   
    # Sort models by the specified metric (descending)
    if sort_by in df_comparison.columns:
        df_comparison = df_comparison.sort_values(by=sort_by, ascending=False)
   
    # Default important metrics if none specified
    if include_metrics is None:
        include_metrics = [
            'combined_metric',     # Overall performance
            'directional_accuracy', # Direction prediction quality
            'f1_score',            # Balance of precision and recall
            'frequency_ratio',     # Predicted vs actual frequency
            'relative_mae'         # Magnitude accuracy (lower is better)
        ]
   
    # Keep only metrics that exist in the dataframe
    include_metrics = [m for m in include_metrics if m in df_comparison.columns]
   
    # SECTION 1: Create a figure with just two subplots side by side
    fig, (ax_ranking, ax_heatmap) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'wspace': 0.3})
   
    # 1. Left: Combined score ranking bar chart
    plot_combined_score_ranking(df_comparison, ax_ranking)
   
    # 2. Right: Performance metrics heatmap
    plot_metrics_heatmap(df_comparison, include_metrics, ax_heatmap)
   
    plt.suptitle("Model Comparison for Large Price Movements", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjusted to give more space for the title
    plt.show()

    # SECTION 2: Second figure with more detailed bar charts
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'hspace': 0.4})
   
    # 5. Detailed metrics bar chart
    plot_detailed_metrics_bars(df_comparison, include_metrics, axes[0])
   
    # 6. Error metrics comparison
    plot_error_comparison(df_comparison, axes[1])
   
    plt.suptitle("Detailed Metrics Comparison", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjusted to give more space for the title
    plt.show()

def plot_combined_score_ranking(df, ax):
    """Plot combined score ranking with improved layout"""
    models = df['model_name'].values if 'model_name' in df.columns else df.index
    scores = df['combined_metric'].values
    
    # Horizontal bar chart for better label visibility
    bars = ax.barh(models, scores, color='skyblue')
    ax.set_title('Model Ranking by Combined Score', fontsize=14, pad=20)
    ax.set_xlabel('Combined Score', fontsize=12)
    
    # Add value annotations with improved positioning and visibility
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(
            bar.get_width() + 0.01,  # Position slightly to the right of the bar
            bar.get_y() + bar.get_height()/2,  # Vertical center of the bar
            f'{score:.3f}',
            va='center',
            fontweight='bold',
            color='black'  # Ensure visibility with black text
        )
    
    # Ensure y-axis labels are fully visible
    plt.setp(ax.get_yticklabels(), fontsize=10)
    ax.tick_params(axis='y', which='major', pad=8)  # Add padding to tick labels
    
    # Add grid for better readability
    ax.grid(axis='x', linestyle='--', alpha=0.7)

def plot_metrics_heatmap(df, metrics, ax):
    """Plot metrics heatmap with improved layout"""
    # Extract just the metrics we need and the model names
    if 'model_name' in df.columns:
        plot_df = df.set_index('model_name')[metrics]
    else:
        plot_df = df[metrics]
    
    # Create heatmap with improved spacing
    im = ax.imshow(plot_df.values, cmap='viridis', aspect='auto')
    
    # Add colorbar with proper size
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Value', rotation=270, labelpad=15)
    
    # Configure axes with improved labels
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(plot_df)))
    
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_yticklabels(plot_df.index, fontsize=10)
    
    # Rotate the tick labels for better visibility
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations with contrasting colors for visibility
    for i in range(len(plot_df)):
        for j in range(len(metrics)):
            value = plot_df.iloc[i, j]
            # Choose text color based on background for better contrast
            text_color = 'white' if value > plot_df.values.mean() else 'black'
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", color=text_color, fontsize=9)
    
    ax.set_title("Performance Metrics Heatmap", fontsize=14, pad=20)

def plot_detailed_metrics_bars(df, metrics, ax):
    """Plot detailed metrics bar chart with improved layout"""
    # Get model names
    models = df['model_name'].values if 'model_name' in df.columns else df.index
    
    # Set width of bars
    width = 0.8 / len(metrics)
    
    # Plot bars for each metric with improved spacing
    for i, metric in enumerate(metrics):
        positions = np.arange(len(models)) + (i - len(metrics)/2 + 0.5) * width
        ax.bar(positions, df[metric].values, width, label=metric)
    
    # Add labels and title with improved spacing
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Detailed Performance Metrics Comparison', fontsize=14, pad=20)
    
    # Set x-ticks at the center of the groups
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, fontsize=10, rotation=45, ha='right')
    
    # Add legend with better positioning - moved higher to avoid overlap
    ax.legend(bbox_to_anchor=(0.5, -0.10), loc='upper center', ncol=len(metrics), fontsize=10)
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)

def plot_error_comparison(df, ax):
    """Plot error metrics comparison with improved layout"""
    # Get model names
    models = df['model_name'].values if 'model_name' in df.columns else df.index
    
    # Error metrics to include
    error_metrics = [col for col in df.columns if 'error' in col.lower() or 'mae' in col.lower() or 'mse' in col.lower()]
    
    if not error_metrics:
        ax.text(0.5, 0.5, 'No error metrics found', ha='center', va='center', fontsize=14)
        return
    
    # Set width of bars
    width = 0.8 / len(error_metrics)
    
    # Plot bars for each error metric with improved spacing
    for i, metric in enumerate(error_metrics):
        positions = np.arange(len(models)) + (i - len(error_metrics)/2 + 0.5) * width
        ax.bar(positions, df[metric].values, width, label=metric)
    
    # Add labels and title with improved spacing
    ax.set_ylabel('Error Value', fontsize=12)
    ax.set_title('Error Metrics Comparison (Lower is Better)', fontsize=14, pad=20)
    
    # Set x-ticks at the center of the groups
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, fontsize=10, rotation=45, ha='right')
    
    # Add legend with better positioning - moved higher to avoid overlap
    ax.legend(bbox_to_anchor=(0.5, -0.10), loc='upper center', ncol=min(len(error_metrics), 4), fontsize=10)
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)