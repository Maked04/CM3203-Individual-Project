import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # type: ignore
from src.data_processing.lstm_data_preprocessing import reduce_time_bucket_features
from src.data_processing.loader import load_time_bucket_data


def generate_data(features_config, time_bucket_folder, test_size):
    # Get the train test data set used to train the model were testing

    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    token_time_buckets, time_bucket_config = load_time_bucket_data(time_bucket_folder)

    token_datasets = []
    for token_address, data in token_time_buckets.items():
        X = data["X"]
        y = data["y"]
        bucket_times = data["bucket_times"]

        # Only get the features listed in features_config
        X = reduce_time_bucket_features(X, features_config)

        token_datasets.append((X, y, token_address, bucket_times))

    # Combine all token data
    all_X = np.vstack([data[0] for data in token_datasets])
    all_y = np.vstack([data[1].reshape(-1, 1) for data in token_datasets])

    # Scale features
    num_samples, time_steps, features = all_X.shape
    X_reshaped = all_X.reshape(num_samples * time_steps, features)
    X_scaled = X_scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(num_samples, time_steps, features)

    # Scale target variable also using StandardScaler to preserve direction
    y_scaled = y_scaler.fit_transform(all_y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=test_size, shuffle=False)

    return X_train, X_test, y_train, y_test, X_scaler, y_scaler

def order_features_config(features_config_dict):
    """
    Orders a features_config dict according to the expected FeaturesConfig fields.
    If a field is missing, it defaults to False.
    """
    feature_keys = [
        "trade_size_ratio",
        "liquidity_ratio",
        "relative_time",
        "absolute_time",
        "price_change",
        "wallet_trade_size_deviation",
        "volume_prior",
        "trade_count_prior",
        "rough_pnl",
        "average_roi",
        "win_rate",
        "average_hold_duration"
    ]
    return [features_config_dict.get(key, False) for key in feature_keys]
