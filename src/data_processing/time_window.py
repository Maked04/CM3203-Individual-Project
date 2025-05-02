import numpy as np

# Create x second time windows of aggregated features


# Take all feature vectors, get rolling windows of 30 seconds to predict whether price doubles or more in the next 30 seconds

class FeatureWindowConfig:
    def __init__(
        self,
        window_size=30,
        prediction_horizon=30,
        price_jump_threshold=2, # Classifies upwards moves of 2x as large
        step_size=None,  # How many seconds to advance when creating the next bucket
    ):
        # Bucket configuration
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        
        # Target variable configuration
        self.price_jump_threshold = price_jump_threshold # Min price jump to classify as a big move (classified as a 1)
        
        # Step size for sliding window (defaults to half the window size)
        if not step_size:
            self.step_size = window_size // 2
        self.step_size = step_size



def get_feature_windows(feature_matrix, timestamps, prices, config: FeatureWindowConfig):
    """
    Creates sliding windows of features from feature matrix for time series prediction.
   
    Args:
        feature_matrix: Matrix containing only the features
        timestamps: Array of timestamps for each feature row
        prices: Array of token prices for each feature row
        config: FeatureWindowConfig object with settings for window creation and target calculation
    """
    X, y = [], []
    bucket_times = [] 

    min_time = np.min(timestamps)
    max_time = np.max(timestamps)
    
    for start_time in np.arange(min_time, max_time, config.step_size):
        end_time = start_time + config.window_size

        # Get indices of times falling in range
        window_indexes = np.where((timestamps >= start_time) & (timestamps < end_time))[0]
        window_tx_features = feature_matrix[window_indexes]

        # Convert all individual tx based features into aggregated window feature vector
        window_feature_vector = generate_window_feature_vector(window_tx_features)

        # Get binary classification target
        horizon_end_time = end_time + config.prediction_horizon
        horizon_indexes = np.where((timestamps >= end_time) & (timestamps < horizon_end_time))[0]
        horizon_prices = prices[horizon_indexes]

        end_window_price = prices[window_indexes[-1]] 

        if max(horizon_prices) / end_window_price > config.price_jump_threshold:
            target_variable = 1  # If price jumped over threshold
        else:
            target_variable = 0

        X.append(window_feature_vector)
        y.append(target_variable)
        bucket_times.append((start_time, end_time))

    return X, y, bucket_times

def generate_window_feature_vector(feature_vectors):
    """
    Generate a single feature vector summarizing a window of transaction feature vectors.
    
    Args:
        feature_vectors: List of feature vectors, each representing a single swap transaction
                         between SOL and an SPL token, ordered by time.
    
    Returns:
        dict: A single feature vector representing the aggregated window statistics
    """
    if not feature_vectors:
        return None
    
    # Initialize aggregation variables
    n_transactions = len(feature_vectors)
    
    # Time-related features
    start_time = feature_vectors[0].absolute_time
    end_time = feature_vectors[-1].absolute_time
    window_duration = end_time - start_time if n_transactions > 1 else 0
    
    # Price-related features
    cumulative_price_change = sum(fv.price_change for fv in feature_vectors)
    price_volatility = np.std([fv.price_change for fv in feature_vectors]) if n_transactions > 1 else 0
    
    # Trade size features
    avg_trade_size_ratio = np.mean([fv.trade_size_ratio for fv in feature_vectors])
    max_trade_size_ratio = max([fv.trade_size_ratio for fv in feature_vectors])
    trade_size_ratio_trend = np.polyfit([i for i in range(n_transactions)], 
                                        [fv.trade_size_ratio for fv in feature_vectors], 
                                        1)[0] if n_transactions > 2 else 0
    
    # Liquidity features
    avg_liquidity_ratio = np.mean([fv.liquidity_ratio for fv in feature_vectors])
    liquidity_ratio_trend = np.polyfit([i for i in range(n_transactions)], 
                                      [fv.liquidity_ratio for fv in feature_vectors], 
                                      1)[0] if n_transactions > 2 else 0
    
    # Wallet behavior features
    avg_wallet_trade_size_deviation = np.mean([fv.wallet_trade_size_deviation for fv in feature_vectors])
    total_volume = sum([fv.volume_prior for fv in feature_vectors])
    total_trade_count = sum([fv.trade_count_prior for fv in feature_vectors])
    avg_rough_pnl = np.mean([fv.rough_pnl for fv in feature_vectors])
    
    # Performance metrics
    weighted_roi = sum([fv.average_roi * fv.trade_size_ratio for fv in feature_vectors]) / sum([fv.trade_size_ratio for fv in feature_vectors]) if sum([fv.trade_size_ratio for fv in feature_vectors]) > 0 else 0
    avg_win_rate = np.mean([fv.win_rate for fv in feature_vectors])
    avg_hold_duration = np.mean([fv.average_hold_duration for fv in feature_vectors])
    
    # Create window feature vector
    window_features = {
        # Time and transaction features
        "window_size": n_transactions,
        "window_duration": window_duration,
        "transaction_frequency": n_transactions / window_duration if window_duration > 0 else 0,
        
        # Price features
        "cumulative_price_change": cumulative_price_change,
        "price_volatility": price_volatility,
        
        # Trade size features
        "avg_trade_size_ratio": avg_trade_size_ratio,
        "max_trade_size_ratio": max_trade_size_ratio,
        "trade_size_ratio_trend": trade_size_ratio_trend,
        
        # Liquidity features
        "avg_liquidity_ratio": avg_liquidity_ratio, 
        "liquidity_ratio_trend": liquidity_ratio_trend,
        
        # Wallet behavior
        "avg_wallet_trade_size_deviation": avg_wallet_trade_size_deviation,
        "total_volume_in_window": total_volume,
        "total_trades_in_window": total_trade_count,
        "avg_rough_pnl": avg_rough_pnl,
        
        # Performance metrics
        "weighted_roi": weighted_roi,
        "avg_win_rate": avg_win_rate,
        "avg_hold_duration": avg_hold_duration,
        
        # First and last transaction times
        "start_time": start_time,
        "end_time": end_time,
    }
    
    return window_features