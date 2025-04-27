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


def plot_multiple_model_results_old(model_comparison):
    # Convert the results into a DataFrame for easy plotting
    df_comparison = pd.DataFrame(model_comparison)

    # Plotting the comparison
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Bar width and x locations
    bar_width = 0.3
    x = np.arange(len(df_comparison))

    # Bar plot for directional accuracy
    ax1.bar(x - bar_width, df_comparison["directional_accuracy"], width=bar_width, label="Directional Accuracy", color="skyblue")

    # Create a second y-axis for large move Average MAE
    ax2 = ax1.twinx()
    ax2.bar(x, df_comparison["relative_mae"], width=bar_width, label="Large Move Relative MAE", color="lightcoral")

    # Create a third y-axis for combined metric
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))  # Move third axis to the right
    ax3.bar(x + bar_width, df_comparison["combined_metric"], width=bar_width, label="Combined Metric", color="lightgreen")

    # Set labels and title
    ax1.set_xlabel("Model", fontsize=12)
    ax1.set_ylabel("Directional Accuracy", color="skyblue", fontsize=12)
    ax2.set_ylabel("Large Move Average MAE", color="lightcoral", fontsize=12)
    ax3.set_ylabel("Combined Metric", color="lightgreen", fontsize=12)
    plt.title("Comparison of Models on Large Move Metrics", fontsize=15)

    # Set x-ticks
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_comparison["model_name"], rotation=45, ha='right')

    # Legends
    lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2, ax3]]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    ax1.legend(lines, labels, loc='upper left')

    # Layout
    plt.tight_layout()
    plt.show()

def plot_multiple_model_results(model_comparison, sort_by='combined_metric', include_metrics=None):
    """
    Create comprehensive visualizations to compare multiple models on large price movement predictions.
    
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
    
    # Create a figure with a complex layout
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.2])
    
    # 1. Top left: Radar/Spider chart for multi-dimensional comparison
    ax_radar = fig.add_subplot(gs[0, 0], polar=True)
    plot_radar_chart(df_comparison, include_metrics, ax_radar)
    
    # 2. Top right: Combined score ranking bar chart
    ax_ranking = fig.add_subplot(gs[0, 1])
    plot_combined_score_ranking(df_comparison, ax_ranking)
    
    # 3. Bottom left: Performance metrics heatmap
    ax_heatmap = fig.add_subplot(gs[1, 0])
    plot_metrics_heatmap(df_comparison, include_metrics, ax_heatmap)
    
    # 4. Bottom right: Precision-Recall scatter with frequency bubbles
    ax_scatter = fig.add_subplot(gs[1, 1])
    plot_precision_recall_frequency(df_comparison, ax_scatter)
    
    plt.suptitle("Comprehensive Model Comparison for Large Price Movements", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
    plt.show()
    
    # Optional: Second figure with more detailed bar charts
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    # 5. Detailed metrics bar chart
    plot_detailed_metrics_bars(df_comparison, include_metrics, axes[0])
    
    # 6. Error metrics comparison
    plot_error_comparison(df_comparison, axes[1])
    
    plt.suptitle("Detailed Metrics Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
    plt.show()

def plot_radar_chart(df, metrics, ax):
    """Plot radar/spider chart comparing models across multiple dimensions."""
    # Normalize metrics for radar chart (0-1 scale)
    df_radar = df.copy()
    
    # For metrics where lower is better, invert the values
    inverse_metrics = ['relative_mae', 'relative_mse']
    for metric in metrics:
        if metric in inverse_metrics and metric in df_radar.columns:
            # Invert and normalize to 0-1 scale
            max_val = df_radar[metric].max()
            min_val = df_radar[metric].min()
            if max_val > min_val:
                df_radar[metric] = 1 - ((df_radar[metric] - min_val) / (max_val - min_val))
            else:
                df_radar[metric] = 1.0
        elif metric in df_radar.columns:
            # Normalize to 0-1 scale
            max_val = df_radar[metric].max()
            min_val = df_radar[metric].min()
            if max_val > min_val:
                df_radar[metric] = (df_radar[metric] - min_val) / (max_val - min_val)
            else:
                df_radar[metric] = 1.0
    
    # Number of metrics (dimensions)
    N = len(metrics)
    
    # Create angles for each metric
    angles = [n / N * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Plot each model
    for i, model in enumerate(df_radar['model_name']):
        values = df_radar.loc[df_radar['model_name'] == model, metrics].values.flatten().tolist()
        values += values[:1]  # Close the loop
        
        # Plot the model line
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
        ax.fill(angles, values, alpha=0.1)
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each metric and label them
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    
    # Draw the y-axis labels (0-1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.set_ylim(0, 1)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    ax.set_title('Multi-dimensional Performance Comparison', fontsize=14)

def plot_combined_score_ranking(df, ax):
    """Plot combined score ranking of models."""
    if 'combined_metric' not in df.columns:
        ax.text(0.5, 0.5, 'Combined metric not available', 
                ha='center', va='center', fontsize=12)
        return
        
    # Sort by combined metric
    df_sorted = df.sort_values('combined_metric', ascending=True)
    
    # Horizontal bar chart
    bars = ax.barh(df_sorted['model_name'], df_sorted['combined_metric'], 
                  color=plt.cm.viridis(np.linspace(0, 0.8, len(df))))
    
    # Add values at the end of bars
    for i, v in enumerate(df_sorted['combined_metric']):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center')
    
    # Add precision and recall as text annotations
    if 'precision' in df.columns and 'recall' in df.columns:
        for i, (_, row) in enumerate(df_sorted.iterrows()):
            p_r_text = f"P: {row['precision']:.2f}, R: {row['recall']:.2f}"
            ax.text(row['combined_metric'] / 2, i, p_r_text, 
                   ha='center', va='center', color='white', fontweight='bold')
    
    ax.set_xlabel('Combined Performance Score')
    ax.set_title('Model Ranking by Combined Score', fontsize=14)
    ax.grid(axis='x', linestyle='--', alpha=0.7)

def plot_metrics_heatmap(df, metrics, ax):
    """Plot heatmap of all metrics for visual comparison."""
    # Select relevant columns
    heatmap_data = df.set_index('model_name')[metrics]
    
    # Normalize data for better visualization
    normalized_data = pd.DataFrame(index=heatmap_data.index)
    
    # For each metric, scale to 0-1 range
    for col in heatmap_data.columns:
        if col in ['relative_mae', 'relative_mse']:  # Metrics where lower is better
            min_val = heatmap_data[col].min()
            max_val = heatmap_data[col].max()
            if max_val > min_val:
                normalized_data[col] = 1 - ((heatmap_data[col] - min_val) / (max_val - min_val))
            else:
                normalized_data[col] = 1.0
        else:  # Metrics where higher is better
            min_val = heatmap_data[col].min()
            max_val = heatmap_data[col].max()
            if max_val > min_val:
                normalized_data[col] = (heatmap_data[col] - min_val) / (max_val - min_val)
            else:
                normalized_data[col] = 1.0
    
    # Create heatmap
    sns.heatmap(normalized_data, annot=heatmap_data.round(3), fmt='.3f', 
                cmap='viridis', linewidths=0.5, ax=ax)
    
    # Improve labels
    ax.set_title('Performance Metrics Heatmap', fontsize=14)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    # Add a note about normalization
    ax.text(0.5, -0.12, 'Note: Colors show normalized values (0-1), numbers show actual metric values', 
            transform=ax.transAxes, ha='center', fontsize=10, style='italic')

def plot_precision_recall_frequency(df, ax):
    """Create a scatter plot with precision vs recall, with bubble size as frequency."""
    if not all(m in df.columns for m in ['precision', 'recall', 'frequency_ratio']):
        ax.text(0.5, 0.5, 'Precision, recall, or frequency metrics not available', 
                ha='center', va='center', fontsize=12)
        return
    
    # Determine bubble size based on frequency ratio
    # Adjust frequency ratio to be centered at 1.0 (perfect)
    bubble_sizes = 100 * np.exp(-0.5 * (df['frequency_ratio'] - 1.0)**2)
    
    # Create scatter plot
    scatter = ax.scatter(df['recall'], df['precision'], s=bubble_sizes, 
                        c=df['combined_metric'], cmap='viridis', 
                        alpha=0.7, edgecolors='w')
    
    # Add model names as labels
    for i, model in enumerate(df['model_name']):
        ax.annotate(model, (df['recall'].iloc[i], df['precision'].iloc[i]),
                   xytext=(5, 5), textcoords='offset points')
    
    # Add reference line for precision = recall
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    
    # Add gridlines
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Set axis limits
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # Add labels and title
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision vs Recall with Frequency Ratio', fontsize=14)
    
    # Add colorbar for combined metric
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Combined Score')
    
    # Add a legend for bubble size
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Perfect (ratio = 1.0)', 
              markerfacecolor='gray', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Good (ratio = 0.5 or 2.0)', 
              markerfacecolor='gray', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Poor (ratio < 0.25 or > 4.0)', 
              markerfacecolor='gray', markersize=4)
    ]
    ax.legend(handles=legend_elements, title='Frequency Ratio', 
             loc='lower right')

def plot_detailed_metrics_bars(df, metrics, ax):
    """Create a detailed bar chart with all metrics side by side."""
    # Melt the dataframe for easier plotting with seaborn
    metrics_to_plot = [m for m in metrics if m in df.columns]
    plot_data = pd.melt(df, id_vars=['model_name'], value_vars=metrics_to_plot,
                       var_name='Metric', value_name='Value')
    
    # Replace underscores with spaces and capitalize metric names
    plot_data['Metric'] = plot_data['Metric'].apply(lambda x: x.replace('_', ' ').title())
    
    # Create grouped bar chart
    sns.barplot(x='model_name', y='Value', hue='Metric', data=plot_data, ax=ax)
    
    # Improve appearance
    ax.set_title('Detailed Metrics Comparison', fontsize=14)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.legend(title='Metric', bbox_to_anchor=(1.01, 1), loc='upper left')
    
    # Rotate x-axis labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

def plot_error_comparison(df, ax):
    """Plot error metrics and ratios for comparing model accuracy."""
    error_metrics = ['relative_mae', 'relative_mse']
    available_errors = [m for m in error_metrics if m in df.columns]
    
    if not available_errors:
        ax.text(0.5, 0.5, 'No error metrics available', 
                ha='center', va='center', fontsize=12)
        return
    
    # Create a dataframe for plotting
    plot_data = pd.melt(df, id_vars=['model_name'], value_vars=available_errors,
                       var_name='Error Metric', value_name='Value')
    
    # Improve metric names
    plot_data['Error Metric'] = plot_data['Error Metric'].apply(
        lambda x: x.replace('relative_mae', 'Relative MAE').replace('relative_mse', 'Relative MSE')
    )
    
    # Create horizontal bar chart
    sns.barplot(x='Value', y='model_name', hue='Error Metric', data=plot_data, 
               orient='h', ax=ax)
    
    # Add vertical line at x=1.0 for reference
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.7)
    
    # Improve appearance
    ax.set_title('Error Metrics Comparison (Lower is Better)', fontsize=14)
    ax.set_xlabel('Error Value', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    ax.legend(title='Error Metric')
    
    # Add grid lines
    ax.grid(axis='x', linestyle='--', alpha=0.7)