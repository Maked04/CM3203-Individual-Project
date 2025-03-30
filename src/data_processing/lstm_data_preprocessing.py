from src.data_processing.loader import load_token_data
from src.data_processing.processor import remove_price_anomalies
import pandas as pd
import numpy as np
import math


import numpy as np

def get_trade_size_ratio(row):
    """Calculate the ratio of trade size to balance after/before trade."""
    if row["bc_spl_after"] > row["bc_spl_before"]:  # Sell
        if row["bc_spl_after"] == 0 or not np.isfinite(row["bc_spl_after"]):
            return None  # Cannot calculate ratio with zero denominator or infinite value
        ratio = abs(row["bc_spl_after"] - row["bc_spl_before"]) / row["bc_spl_after"]
        return None if not np.isfinite(ratio) else ratio
    elif row["bc_spl_after"] < row["bc_spl_before"]:  # Buy
        if row["bc_spl_before"] == 0 or not np.isfinite(row["bc_spl_before"]):
            return None  # Cannot calculate ratio with zero denominator or infinite value
        ratio = abs(row["bc_spl_after"] - row["bc_spl_before"]) / row["bc_spl_before"]
        return None if not np.isfinite(ratio) else ratio
    return 0  # No change

def get_trade_liquidity_ratio(row):
    """Calculate the ratio of SOL balance to SPL balance before trade."""
    if row["bc_spl_before"] == 0 or not np.isfinite(row["bc_spl_before"]) or not np.isfinite(row["bc_sol_before"]):
        return None  # Cannot calculate ratio with zero denominator or infinite values
    ratio = row["bc_sol_before"] / row["bc_spl_before"]
    return None if not np.isfinite(ratio) else ratio

def get_token_features(token_address, relative_time=True, min_sol_size=0.1):
    """Generate feature matrix for token transactions."""
    # Load and clean data
    df = load_token_data(token_address)
    cleaned_df = remove_price_anomalies(df)
    
    # Compute price changes
    price_changes = cleaned_df["token_price"].pct_change().iloc[1:].values  # Convert to array
    
    # Feature matrix
    feature_matrix = []
    
    # Iterate over the dataframe rows
    previous_time = cleaned_df.iloc[0]["slot"]  # Initialize previous time with the first transaction's time
    for i, (_, row) in enumerate(cleaned_df.iloc[1:].iterrows()):  # Ensure sequential iteration
        # Skip small transactions
        if row["bc_sol_before"] - row["bc_sol_after"] < min_sol_size:
            continue
            
        # Get price change
        price_change = price_changes[i]  # Use proper indexing from numpy array
        
        # Calculate time difference
        time = row["slot"] - previous_time
        
        # Get feature values
        trade_size_ratio = get_trade_size_ratio(row)
        liquidity_ratio = get_trade_liquidity_ratio(row)
        
        # Check for valid values (not None, not NaN, and finite)
        if (price_change is not None and not np.isnan(price_change) and np.isfinite(price_change) and 
            trade_size_ratio is not None and np.isfinite(trade_size_ratio) and 
            liquidity_ratio is not None and np.isfinite(liquidity_ratio) and
            np.isfinite(time)):
            
            feature_matrix.append([
                row.name,
                row["token_price"],
                trade_size_ratio,
                liquidity_ratio,
                time,
                price_change
            ])
            
            if relative_time:
                previous_time = row["slot"]  # Update previous time for relative time calculation
    
    return np.array(feature_matrix)


def get_time_buckets(feature_matrix, bucket_size=30, prediction_horizon=1, min_txs_per_second=1, max_multiple=True, cumulative_price_change=False):
    """
    Creates sliding windows of time buckets from feature matrix for time series prediction.
   
    Args:
        feature_matrix: Matrix containing features (last column is price_change from pct_change())
        bucket_size: Number of seconds per time bucket
        prediction_horizon: Number of buckets to use for prediction
        min_txs_per_second: Minimum number of txs per second a bucket can have
        max_multiple: Whether to use max multiple in prediction horizon for the target value
        cumulative_price_change: Whether to use cumulative price change across prediction horizon for the target value
       
    Returns:
        X: Input sequences as standard list as they may have different lengths so cant be numpy array
        y: Max multiples or cumulative price change in prediction horizon as numpy array
        bucket_times: List of tuples containing (start_time, end_time) for each bucket
    """
    X, y = [], []
    bucket_times = []  # List to store start and end times of each bucket
    
    feature_times = feature_matrix[:, 0]  # Assuming block_time is the first column
    feature_prices = feature_matrix[:, 1]  # Assuming token_price is the second column
    feature_matrix = feature_matrix[:, 2:]  # Remove the time and token price columns (Only used for target calculation)
    
    min_time = np.min(feature_times)
    max_time = np.max(feature_times)
    
    for start_time in np.arange(min_time, max_time, 1):
        end_time = start_time + bucket_size
        # Get indicies of times falling in range
        bucket_indexes = np.where((feature_times >= start_time) & (feature_times < end_time))[0]
        bucket_features = feature_matrix[bucket_indexes]
        
        if len(bucket_features) / bucket_size < min_txs_per_second:
            continue
            
        # Get target variable
        target_variable = None
        horizon_indexes = np.where((feature_times >= end_time) & (feature_times < (end_time + (bucket_size * prediction_horizon))))[0]
        
        if max_multiple:
            if len(bucket_indexes) == 0:
                continue
            price_at_end = feature_prices[bucket_indexes[-1]] # Price at end of bucket
            horizon_prices = feature_prices[horizon_indexes]
            if horizon_prices.size > 0:
                max_upside = max(horizon_prices) / price_at_end - 1
                max_downside = min(horizon_prices) / price_at_end - 1
                target_variable = max_upside if abs(max_upside) > abs(max_downside) else max_downside   # Max move
        elif cumulative_price_change:
            horizon_price_chnages = feature_matrix[horizon_indexes, -1]
            target_variable = np.prod(1 + horizon_price_chnages) - 1  # Cumulative change
       
        if target_variable is not None and not np.isnan(target_variable) and not np.isinf(target_variable):
            X.append(bucket_features)
            y.append(target_variable)
            bucket_times.append((start_time, end_time))  # Store the start and end time of this bucket
    
    y = np.array(y)
    return X, y, bucket_times

def pad_sequences_with_price_importance(X_list, max_seq_length=300):
    """
    Pads sequences to a fixed length or truncates them based on price impact importance.
   
    Args:
        X_list: List of feature arrays with variable lengths
        max_seq_length: Maximum sequence length to pad/truncate to
   
    Returns:
        X_padded: Numpy array with shape (n_samples, max_seq_length, n_features)
    """
    if not X_list:
        return np.array([])
   
    n_samples = len(X_list)
    n_features = X_list[0].shape[1]
   
    # Initialize padded array with zeros
    X_padded = np.zeros((n_samples, max_seq_length, n_features))
   
    for i, sequence in enumerate(X_list):
        if len(sequence) <= max_seq_length:
            # If sequence is shorter than max_length, use left padding
            # Place sequence at the end of the padded array
            start_idx = max_seq_length - len(sequence)
            X_padded[i, start_idx:, :] = sequence
        else:
            # If sequence is longer, keep transactions with largest price changes
            # Assuming the last column is price_change
            price_change_col = sequence.shape[1] - 1
           
            # Calculate absolute price changes for importance
            abs_price_changes = np.abs(sequence[:, price_change_col])
           
            # Get indices of transactions sorted by importance (largest price change first)
            important_indices = np.argsort(abs_price_changes)[::-1][:max_seq_length]
           
            # Sort these indices to maintain temporal order
            important_indices = np.sort(important_indices)
           
            # Select the most important transactions while preserving order
            important_transactions = sequence[important_indices]
           
            # Fill the padded array (with left padding if needed)
            start_idx = max_seq_length - len(important_transactions)
            X_padded[i, start_idx:, :] = important_transactions
   
    return X_padded
        

def get_sliding_windows(feature_matrix, sequence_length=10, prediction_horizon=1, time_horizon=False):
    """
    Creates sliding windows from feature matrix for time series prediction.
    
    Args:
        feature_matrix: Matrix containing features (last column is price_change from pct_change())
        sequence_length: Number of past transactions used as input
        prediction_horizon: Number of future transactions or seconds of transactions to calculate cumulative price change
        time_horizon: Whether to use number number of seconds for prediction horizon, false uses number of txs
        
    Returns:
        X: Input sequences of shape (num_samples, sequence_length, num_features)
        y: Cumulative price changes over the prediction horizon of shape (num_samples,)
    """
    X, y = [], []

    feature_times = feature_matrix[:, 0]  # Assuming block_time is the first column
    feature_matrix = feature_matrix[:, 2:]  # Remove the time and token price column    
    
    # Make sure we have enough data for both the sequence and the future prediction
    for i in range(len(feature_matrix) - sequence_length - prediction_horizon + 1):
        # Input sequence: last sequence_length transactions
        X.append(feature_matrix[i : i + sequence_length])
        
        # Target: cumulative price change over the prediction_horizon transactions
        # We collect all price changes in the horizon window
        if time_horizon:
            pred_start_time = feature_times[i + sequence_length]
            pred_end_time = pred_start_time + prediction_horizon
            horizon_indexes = np.where((feature_times >= pred_start_time) & (feature_times <= pred_end_time))[0]
            filtered_indexes = horizon_indexes[horizon_indexes > i + sequence_length] # Remove indexes that are in input sequence
            future_changes = feature_matrix[filtered_indexes, -1]
            
        else:
            future_changes = feature_matrix[i + sequence_length : i + sequence_length + prediction_horizon, -1]
        
        # Calculate cumulative price change (proper way to combine percentage changes)
        # (1 + r1) * (1 + r2) * ... * (1 + rn) - 1
        cumulative_change = np.prod(1 + future_changes) - 1

        # Add to training data only if the result is valid
        if not np.isnan(cumulative_change) and not np.isinf(cumulative_change):
            y.append(cumulative_change)
            # Keep the corresponding X entry (we already added it above)
        else:
            # Remove the X entry if y is invalid
            X.pop()
                
    X = np.array(X)  # Shape: (num_samples, sequence_length, num_features)
    y = np.array(y)  # Shape: (num_samples,)
    
    return X, y


if __name__ == "__main__":
    token_address = "WUb891xiehvvaDURF1r5ZBcbKULPQHcSnNtLfywpump"
    feature_matrix = get_token_features(token_address, relative_time=False)

    print(len(feature_matrix))

    #X, y = get_sliding_windows(feature_matrix, sequence_length=10, prediction_horizon=40, time_horizon=True) 
    X, y = get_time_buckets(feature_matrix, min_txs_per_second=1)
