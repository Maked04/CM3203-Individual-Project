import numpy as np
import os
import datetime
import json
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


# Save model and config (updated to include hyperparameters)
def save_model_with_config(model, tuner, features_config, time_bucket_folder, test_size, early_stopping, X_train, y_train, epochs, batch_size):
    # Timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a base directory
    base_dir = "trained_models"
    os.makedirs(base_dir, exist_ok=True)
    
    # Naming and directory
    model_name = f"lstm_{len(os.listdir(base_dir)) + 1}"
    model_dir = os.path.join(base_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Save the Keras model
    model_path = os.path.join(model_dir, "model.keras")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # 2. Save configuration parameters
    optimizer_name = model.optimizer.__class__.__name__.lower()
    loss_name = model.loss.__name__ if callable(model.loss) else model.loss
    
    model_layers = []
    for layer in model.layers:
        layer_config = {
            "name": layer.name,
            "type": layer.__class__.__name__
        }
        
        if hasattr(layer, "units"):
            layer_config["units"] = layer.units
        
        if hasattr(layer, "activation") and layer.activation is not None:
            if hasattr(layer.activation, "__name__"):
                layer_config["activation"] = layer.activation.__name__
            else:
                layer_config["activation"] = str(layer.activation)
                
        if hasattr(layer, "rate"):
            layer_config["rate"] = layer.rate
        
        try:
            if hasattr(layer, "output") and layer.output is not None:
                output_shape = layer.output.shape.as_list()
                layer_config["output_shape"] = [dim if dim is not None else -1 for dim in output_shape]
        except (AttributeError, ValueError):
            pass
        
        model_layers.append(layer_config)
    
    # --- Add this: get best hyperparameters ---
    best_hps = tuner.get_best_hyperparameters(1)[0]
    best_hyperparameters = best_hps.values
    
    # --- Full config dictionary ---
    config = {
        "features_config": vars(features_config),  # Convert class to dict
        "time_bucket_folder": time_bucket_folder,
        "test_size": test_size,
        "training_params": {
            "optimizer": optimizer_name,
            "loss": loss_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "validation_split": None,
            "early_stopping": {
                "monitor": early_stopping.monitor,
                "patience": early_stopping.patience,
                "min_delta": early_stopping.min_delta,
                "mode": early_stopping.mode,
                "restore_best_weights": early_stopping.restore_best_weights
            }
        },
        "model_architecture": {
            "layers": model_layers,
            "total_params": model.count_params()
        },
        "best_hyperparameters": best_hyperparameters,  # <<< New: save best found HParams
        "timestamp": timestamp,
        "input_shape": [dim if dim is not None else -1 for dim in model.input_shape],
        "X_train_shape": list(X_train.shape),
        "y_train_shape": list(y_train.shape)
    }
    
    # Save config
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    print(f"Configuration saved to {config_path}")
    print(f"\nAll model artifacts saved to {model_dir}")

