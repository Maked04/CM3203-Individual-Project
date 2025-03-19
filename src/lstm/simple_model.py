import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore

from src.data_processing.lstm_data_preprocessing import get_token_features, get_sliding_windows
from src.data_processing.loader import load_token_data
from src.data_processing.processor import remove_price_anomalies

token_address = "2a97apseXcvY4768xkkN2mKsDjtmWJq9DcfR4PsCpump"
# Load data
df = load_token_data(token_address)
    
# Clean data for price features
cleaned_df = remove_price_anomalies(df)

feature_matrix = get_token_features(token_address)
feature_matrix = feature_matrix[0: 30]

X, y = get_sliding_windows(feature_matrix)

# Normalize the features (important for LSTM models)
# Apply scaling only to the features (not the targets)
feature_scaler = MinMaxScaler(feature_range=(0, 1))

# Reshape X to 2D for scaling then back to 3D
num_samples, time_steps, features = X.shape
X_reshaped = X.reshape(num_samples * time_steps, features)
X_scaled = feature_scaler.fit_transform(X_reshaped)
X_scaled = X_scaled.reshape(num_samples, time_steps, features)

# Scale the target variable y
y_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaled = y_scaler.fit_transform(y.reshape(-1, 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, 
               input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))  # Output layer - predicting the price change

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Make predictions
y_pred = model.predict(X_test)

# Inverse transform to get real values
y_pred_actual = y_scaler.inverse_transform(y_pred)
y_test_actual = y_scaler.inverse_transform(y_test)

# Evaluate the model
mse = np.mean((y_pred_actual - y_test_actual) ** 2)
print(f"Mean Squared Error: {mse}")