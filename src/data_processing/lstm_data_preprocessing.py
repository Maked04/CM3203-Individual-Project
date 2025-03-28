from src.data_processing.loader import load_token_data
from src.data_processing.processor import remove_price_anomalies
import pandas as pd
import numpy as np
import math


def get_trade_size_ratio(row):
    if row["bc_spl_after"] > row["bc_spl_before"]:  # Sell
        return abs(row["bc_spl_after"] - row["bc_spl_before"]) / row["bc_spl_after"]
    elif row["bc_spl_after"] < row["bc_spl_before"]:  # Buy
        return abs(row["bc_spl_after"] - row["bc_spl_before"]) / row["bc_spl_before"]
    return 0  # No change

def get_trade_liquidity_ratio(row):
    return row["bc_sol_before"] / row["bc_spl_before"]

def get_token_features(token_address):
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
        price_change = price_changes[i]  # Use proper indexing from numpy array
        if not np.isnan(price_change):  # Ignore NaN values
            feature_matrix.append([
                get_trade_size_ratio(row),
                get_trade_liquidity_ratio(row),
                row["slot"] - previous_time,  # Time difference relative to previous transaction
                price_change
            ])
            previous_time = row["slot"]  # Update previous time to the current transaction's time

    return np.array(feature_matrix)



def get_token_features_basic(token_address):
    # Load and clean data
    df = load_token_data(token_address)
    cleaned_df = remove_price_anomalies(df)

    # Compute price changes
    price_changes = cleaned_df["token_price"].pct_change().iloc[1:].values  # Convert to array

    # Get start time
    start_time = cleaned_df.iloc[0]["slot"]

    # Feature matrix
    feature_matrix = []

    for i, (_, row) in enumerate(cleaned_df.iloc[1:].iterrows()):  # Ensure sequential iteration
        price_change = price_changes[i]  # Use proper indexing from numpy array
        if not np.isnan(price_change):  # Ignore NaN values
            feature_matrix.append([
                row["slot"] - start_time,  # Time difference
                price_change
            ])

    return np.array(feature_matrix)


def get_sliding_windows(feature_matrix, sequence_length=10, prediction_horizon=1):
    """
    Creates sliding windows from feature matrix for time series prediction.
    
    Args:
        feature_matrix: Matrix containing features (last column is price_change from pct_change())
        sequence_length: Number of past transactions used as input
        prediction_horizon: Number of future transactions to calculate cumulative price change
        
    Returns:
        X: Input sequences of shape (num_samples, sequence_length, num_features)
        y: Cumulative price changes over the prediction horizon of shape (num_samples,)
    """
    X, y = [], []
    
    # Make sure we have enough data for both the sequence and the future prediction
    for i in range(len(feature_matrix) - sequence_length - prediction_horizon + 1):
        # Input sequence: last sequence_length transactions
        X.append(feature_matrix[i : i + sequence_length])
        
        # Target: cumulative price change over the prediction_horizon transactions
        # We collect all price changes in the horizon window
        future_changes = feature_matrix[i + sequence_length : i + sequence_length + prediction_horizon, -1]
        print(f"Using these changes: {future_changes}")
        
        # Calculate cumulative price change (proper way to combine percentage changes)
        # (1 + r1) * (1 + r2) * ... * (1 + rn) - 1
        cumulative_change = np.prod(1 + future_changes) - 1
        print("We get cumulative change of: ", cumulative_change)

        print("vs adding way")
        print("We get cumulative change of: ", sum(future_changes))

        #exit()
        
        y.append(cumulative_change)
        
    X = np.array(X)  # Shape: (num_samples, sequence_length, num_features)
    y = np.array(y)  # Shape: (num_samples,)
    
    return X, y

def save_data(token_addresses, output_folder="../../data/"):
    all_features = []
    skipped_tokens = []
    
    for token in token_addresses:
        feature_vectors = get_token_features(token)


if __name__ == "__main__":
    token_address = "WUb891xiehvvaDURF1r5ZBcbKULPQHcSnNtLfywpump"
    # Load data
    df = load_token_data(token_address)
        
    # Clean data for price features
    cleaned_df = remove_price_anomalies(df)

    cleaned_df = cleaned_df[0: 20]

    feature_matrix = get_token_features(token_address)

    X, y = get_sliding_windows(feature_matrix, sequence_length=10, prediction_horizon=2) 