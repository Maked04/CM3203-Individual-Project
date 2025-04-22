from src.data_processing.loader import load_token_data, get_metric_by_tx_sig, get_data_dir, load_data_file
from src.data_processing.processor import remove_price_anomalies
import numpy as np
import datetime
import os
import json

class FeaturesConfig:
    def __init__(
        self,
        trade_size_ratio=False,
        liquidity_ratio=False,
        relative_time=False,
        absolute_time=False,
        price_change=False,
        wallet_trade_size_deviation=False,
        volume_prior=False,
        trade_count_prior=False,
        rough_pnl=False,
        average_roi=False,
        win_rate=False,
        average_hold_duration=False
    ):
        # Standard features
        self.trade_size_ratio = trade_size_ratio
        self.liquidity_ratio = liquidity_ratio
        self.relative_time = relative_time
        self.absolute_time = absolute_time
        self.price_change = price_change
        # Wallet specific features
        self.wallet_trade_size_deviation = wallet_trade_size_deviation
        self.volume_prior = volume_prior
        self.trade_count_prior = trade_count_prior
        self.rough_pnl = rough_pnl
        self.average_roi = average_roi
        self.win_rate = win_rate
        self.average_hold_duration = average_hold_duration


class TimeBucketConfig:
    def __init__(
        self,
        bucket_size=30,
        prediction_horizon=1,
        min_txs_per_second=1,
        use_max_multiple=True,
        use_cumulative_price_change=False,
        step_size=1,  # How many seconds to advance when creating the next bucket
        max_seq_length=300
    ):
        # Bucket configuration
        self.bucket_size = bucket_size
        self.prediction_horizon = prediction_horizon
        self.min_txs_per_second = min_txs_per_second
        
        # Target variable configuration
        self.use_max_multiple = use_max_multiple
        self.use_cumulative_price_change = use_cumulative_price_change
        
        # Step size for sliding window (defaults to 1 second)
        self.step_size = step_size

        # Max sequence length per time buclet
        self.max_seq_length = max_seq_length
        
        # Validate configuration
        if self.use_max_multiple and self.use_cumulative_price_change:
            raise ValueError("Cannot use both max_multiple and cumulative_price_change for target calculation")
        

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

def validate_features(features):
    for feature in features:
        if feature is None:
            return False
        if not isinstance(feature, (int, float, np.number)):
            return False
        if not np.isfinite(feature):  # This includes both inf and nan
            return False
    return True


def get_token_features_and_metadata(token_address, min_sol_size=0.1):
    """
    Extract features from token data along with metadata needed for target creation.
    
    Returns:
        feature_matrix: Matrix containing only the features
        timestamps: Array of timestamps for each row
        prices: Array of token prices for each row
    """
    df = load_token_data(token_address)
    cleaned_df = remove_price_anomalies(df)

    # Compute price changes
    price_changes = cleaned_df["token_price"].pct_change().iloc[1:].values  # Convert to array
    
    start_time = cleaned_df.iloc[0]["slot"]  # Initialize start time with the first transaction's time
    previous_time = cleaned_df.iloc[0]["slot"]  # Initialize previous time with the first transaction's time

    # Feature matrix
    all_features = []
    timestamps = []
    prices = []
    for i, (_, row) in enumerate(cleaned_df.iloc[1:].iterrows()): 
        if abs(row["bc_sol_before"] - row["bc_sol_after"]) < min_sol_size:
            continue

        wallet_metrics = get_metric_by_tx_sig(row["tx_sig"])
        if wallet_metrics is None:
            #continue
            wallet_metrics = {"trade_size_deviation": 1,
                              "volume_prior": 1,
                              "trade_count_prior": 1,
                              "rough_pnl": 1,
                              "average_roi": 1,
                              "win_rate": 1,
                              "average_hold_duration": 1}

        relative_time = row["slot"] - previous_time
        absolute_time = row["slot"] - start_time

        previous_time = row["slot"]  # Update previous time for relative time calculation

        # IMPORTANT, order of features must match the order of variables in features config
        features = [get_trade_size_ratio(row),
                    get_trade_liquidity_ratio(row),
                    relative_time,
                    absolute_time,
                    price_changes[i],
                    wallet_metrics["trade_size_deviation"],
                    wallet_metrics["volume_prior"],
                    wallet_metrics["trade_count_prior"],
                    wallet_metrics["rough_pnl"],
                    wallet_metrics["average_roi"],
                    wallet_metrics["win_rate"],
                    wallet_metrics["average_hold_duration"],
                    ]
        
        if validate_features(features):
            all_features.append(features)

            # Store metadata for target calculation separately
            timestamps.append(row.name)
            prices.append(row["token_price"])

    # Return separate arrays for features and metadata
    return np.array(all_features), np.array(timestamps), np.array(prices)


def get_time_buckets(feature_matrix, timestamps, prices, config: TimeBucketConfig):
    """
    Creates sliding windows of time buckets from feature matrix for time series prediction.
   
    Args:
        feature_matrix: Matrix containing only the features
        timestamps: Array of timestamps for each feature row
        prices: Array of token prices for each feature row
        config: TimeBucketConfig object with settings for bucket creation and target calculation
       
    Returns:
        X: Input sequences as standard list as they may have different lengths so cant be numpy array
        y: Target values calculated according to the configuration
        bucket_times: List of tuples containing (start_time, end_time) for each bucket
    """
    X, y = [], []
    bucket_times = []  # List to store start and end times of each bucket
    
    min_time = np.min(timestamps)
    max_time = np.max(timestamps)
    
    for start_time in np.arange(min_time, max_time, config.step_size):
        end_time = start_time + config.bucket_size
        
        # Get indices of times falling in range
        bucket_indexes = np.where((timestamps >= start_time) & (timestamps < end_time))[0]
        bucket_features = feature_matrix[bucket_indexes]
        
        if len(bucket_features) / config.bucket_size < config.min_txs_per_second:
            continue
            
        # Get target variable
        target_variable = None
        horizon_end_time = end_time + (config.bucket_size * config.prediction_horizon)
        horizon_indexes = np.where((timestamps >= end_time) & (timestamps < horizon_end_time))[0]
        
        if config.use_max_multiple:
            if len(bucket_indexes) == 0:
                continue
            price_at_end = prices[bucket_indexes[-1]]  # Price at end of bucket
            horizon_prices = prices[horizon_indexes]
            if horizon_prices.size > 0:
                max_upside = max(horizon_prices) / price_at_end - 1
                max_downside = min(horizon_prices) / price_at_end - 1
                target_variable = max_upside if abs(max_upside) > abs(max_downside) else max_downside  # Max move
        elif config.use_cumulative_price_change:
            price_changes = np.diff(prices[horizon_indexes]) / prices[horizon_indexes[:-1]] if len(horizon_indexes) > 1 else np.array([])
            target_variable = np.prod(1 + price_changes) - 1 if len(price_changes) > 0 else None  # Cumulative change
       
        if target_variable is not None and not np.isnan(target_variable) and not np.isinf(target_variable):
            X.append(bucket_features)
            y.append(target_variable)
            bucket_times.append((start_time, end_time))  # Store the start and end time of this bucket
    
    X = pad_sequences_with_price_importance(X, config.max_seq_length)
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

def reduce_time_bucket_features(X: np.ndarray, config: FeaturesConfig) -> np.ndarray:
    # Assuming config.__dict__ contains a dictionary with the feature mask as boolean values
    features_mask = list(config.__dict__.values())
    
    # Filter the features based on the mask
    filtered = X[:, :, features_mask]
    
    return filtered

def pre_process_train_test_data(time_bucket_config, tokens_file="cluster_2_tokens.txt"):
    # Create train test data folder if it doesnt exist
    base_dir = os.path.join(get_data_dir(), "time_bucket_data")
    os.makedirs(base_dir, exist_ok=True)

    # Name of new directory is time_bucket_num where num increases per folder
    folder_name = f"time_bucket_{len(os.listdir(base_dir)) + 1}"
    time_bucket_dir = os.path.join(base_dir, folder_name)
    os.makedirs(time_bucket_dir, exist_ok=True)

    config_json = vars(time_bucket_config)

    # Save time bucket config
    with open(os.path.join(time_bucket_dir, "config.json"), "w") as f:
        json.dump(config_json, f, indent=4)

    # Read cluster 2 tokens
    with load_data_file(tokens_file) as f:
        token_addresses = f.read().splitlines()[:-1]

    # Get features
    for token_address in token_addresses:
        features, timestamps, prices = get_token_features_and_metadata(token_address)

        # Get time buckets
        X, y, bucket_times = get_time_buckets(features, timestamps, prices, time_bucket_config) 
        if len(X) == 0 or len(y) == 0 or len(X) != len(y):  # Make sure we have data
            continue
        
        token_dir = os.path.join(time_bucket_dir, token_address)
        os.makedirs(token_dir, exist_ok=True)
        # Save X, y, bucket_times per token
        np.save(os.path.join(token_dir, "X"), X)
        np.save(os.path.join(token_dir, "y"), y)
        np.save(os.path.join(token_dir, "bucket_times"), bucket_times)

def pre_process_train_test_data_filtered(time_bucket_config, tokens_file="cluster_2_tokens.txt", ignore_start_secs=0):
    # Create train test data folder if it doesnt exist
    base_dir = os.path.join(get_data_dir(), "time_bucket_data")
    os.makedirs(base_dir, exist_ok=True)

    # Name of new directory is time_bucket_num where num increases per folder
    folder_name = f"time_bucket_{len(os.listdir(base_dir)) + 1}"
    time_bucket_dir = os.path.join(base_dir, folder_name)
    os.makedirs(time_bucket_dir, exist_ok=True)

    config_json = vars(time_bucket_config)

    # Save time bucket config
    with open(os.path.join(time_bucket_dir, "config.json"), "w") as f:
        json.dump(config_json, f, indent=4)

    # Read cluster 2 tokens
    with load_data_file(tokens_file) as f:
        token_addresses = f.read().splitlines()[:-1]

    # Get features
    for token_address in token_addresses:
        features, timestamps, prices = get_token_features_and_metadata(token_address)

        if ignore_start_secs > 0:
            filtered_indexes = timestamps[timestamps > timestamps[0] + ignore_start_secs]
            features, timestamps, prices = features[filtered_indexes], timestamps[filtered_indexes], prices[filtered_indexes]

        # Get time buckets
        X, y, bucket_times = get_time_buckets(features, timestamps, prices, time_bucket_config) 
        if len(X) == 0 or len(y) == 0 or len(X) != len(y):  # Make sure we have data
            continue
        
        token_dir = os.path.join(time_bucket_dir, token_address)
        os.makedirs(token_dir, exist_ok=True)
        # Save X, y, bucket_times per token
        np.save(os.path.join(token_dir, "X"), X)
        np.save(os.path.join(token_dir, "y"), y)
        np.save(os.path.join(token_dir, "bucket_times"), bucket_times)


def test_saving():
    time_bucket_config = TimeBucketConfig(
    bucket_size=30,  
    prediction_horizon=1,
    min_txs_per_second=1,
    use_max_multiple=True,
    step_size=1,
    max_seq_length=300
    )

    pre_process_train_test_data(time_bucket_config)


def test_reducing_features():
    token_address = "BjHTDNRjKxEhQkxtjo6iTNPdnpcwXFNdnd6SNhWopump"
    min_sol_size = 0.1
    features, timestamps, prices = get_token_features_and_metadata(token_address, min_sol_size=min_sol_size)
    time_bucket_config = TimeBucketConfig(
    bucket_size=30,  
    prediction_horizon=1,
    min_txs_per_second=1,
    use_max_multiple=True,
    step_size=1,
    max_seq_length=300
    )
    X, y, bucket_times = get_time_buckets(features, timestamps, prices, time_bucket_config)

    # X currently contains all features
    features_config = FeaturesConfig(
        trade_size_ratio=True,
        liquidity_ratio=True,
        relative_time=True,
    )

    reduced_X = reduce_time_bucket_features(X, features_config)

if __name__ == "__main__":
    test_saving()