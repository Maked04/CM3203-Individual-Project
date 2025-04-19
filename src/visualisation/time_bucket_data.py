# Import packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# Local imports
from src.data_processing.lstm_data_preprocessing import reduce_time_bucket_features
from src.data_processing.loader import load_time_bucket_data


def load_data(time_bucket_folder = "time_bucket_1"):
    # Change based on which time bucket configuration you want to used, their preprocessed in different folders
    token_time_buckets, time_bucket_config = load_time_bucket_data(time_bucket_folder)

    token_datasets = []
    for token_address, data in token_time_buckets.items():
        X = data["X"]
        y = data["y"]
        bucket_times = data["bucket_times"]

        token_datasets.append((X, y, token_address, bucket_times))

    # Combine all token data
    all_X = np.vstack([data[0] for data in token_datasets])
    all_y = np.vstack([data[1].reshape(-1, 1) for data in token_datasets])

    return all_X, all_y

def plot_target_distribution(y, cutoff=0.5):
    large_y = y[abs(y) > cutoff]
    plt.hist(large_y, bins=100, alpha=0.5, label="Target Data")
    plt.legend()
    plt.title("Distribution of Target (y)")
    plt.show()


def scale_data(X, y):
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    # Scale features
    num_samples, time_steps, features = X.shape
    X_reshaped = X.reshape(num_samples * time_steps, features)
    X_scaled = X_scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(num_samples, time_steps, features)

    # Scale target variable also using StandardScaler to preserve direction
    y_scaled = y_scaler.fit_transform(y)

    return X_scaled, y_scaled, X_scaler, y_scaler


def main():
    all_X, all_y = load_data()

    plot_target_distribution(all_y)


if __name__ == "__main__":
    main()