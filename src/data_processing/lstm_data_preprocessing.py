from src.data_processing.loader import load_token_data
from src.data_processing.processor import remove_price_anomalies
import pandas as pd
import numpy as np
import math


# Given list of transactions

# [tx1 tx2 tx3 tx4 tx5 tx6 tx7 tx8 tx9 tx10 tx11 tx12 tx13 tx14 tx15]

# Input is [10 x 4] array

# Input 
# tx2: price change from tx1 to tx2
#      time
#      liquidity ratio
#      trade size ratio

# tx3: price change from tx2 to tx3
#      time
#      liquidity ratio
#      trade size ratio


# .....

#tx11: price change from tx10 to tx11
#      time
#      liquidity ratio
#      trade size ratio

# Target 
# Price change from tx11 to tx12


def get_token_price_changes(token_data):
    return token_data["token_price"].pct_change()

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

    # Get start time
    start_time = cleaned_df.iloc[0]["slot"]

    # Feature matrix
    feature_matrix = []

    for i, (_, row) in enumerate(cleaned_df.iloc[1:].iterrows()):  # Ensure sequential iteration
        price_change = price_changes[i]  # Use proper indexing from numpy array
        if not np.isnan(price_change):  # Ignore NaN values
            feature_matrix.append([
                get_trade_size_ratio(row),
                get_trade_liquidity_ratio(row),
                row["slot"] - start_time,  # Time difference
                price_change
            ])

    return np.array(feature_matrix)


def get_sliding_windows(feature_matrix, sequence_length=10):
    # sequence_length: Number of past transactions used as input
    X, y = [], []

    for i in range(len(feature_matrix) - sequence_length):
        X.append(feature_matrix[i : i + sequence_length])  # Last 10 transactions as input
        y.append(feature_matrix[i + sequence_length, 3])  # Predict next price_change

    X = np.array(X)  # Shape: (num_samples, 10, 4)
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

    X, y = get_sliding_windows(feature_matrix)    
