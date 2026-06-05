# main_DL.py
# ------------------------------------------------------
# Main script for training and evaluating deep learning models
# Author: Guillem Hernández Guillamet
# Version: 1.0
# Date: 04/06/2025
# Description:
#   This script loads time series data, prepares it for model training,
#   selects and trains GRU/LSTM models, evaluates performance, and
#   visualizes the results. Uses utility functions from utils_DL.py.
# ------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from utils_DL import create_model_gru, create_model_lstm, fit_model, prediction, inverse_transform, auto_grid_search

# --- Load and preprocess dataset ---
df = pd.read_csv('your_timeseries_data.csv', index_col=0, parse_dates=True)
target_col = 'target'  # Replace with actual target column name

# Scale features
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(data_scaled, index=df.index, columns=df.columns)

# Create lagged features (example)
def create_lagged_features(df, target, n_lags=5):
    lagged_df = pd.DataFrame(index=df.index)
    for i in range(n_lags):
        lagged_df[f'{target}_lag_{i+1}'] = df[target].shift(i+1)
    lagged_df[target] = df[target]
    return lagged_df.dropna()

supervised_df = create_lagged_features(df_scaled, target_col)
X = supervised_df.drop(target_col, axis=1).values
y = supervised_df[target_col].values

# Reshape for RNN
X = X.reshape((X.shape[0], X.shape[1], 1))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
validation_data = (X_test, y_test)

# --- Train and evaluate GRU ---
print("\n>>> Training GRU")
gru_model = create_model_gru(X_train)
history_gru = fit_model(gru_model, X_train, y_train, epochs=50, batch_size=32, validation_data=validation_data)
yhat_gru = prediction(gru_model, X_test)
y_true_inv_gru, yhat_inv_gru = inverse_transform(y_test, yhat_gru, scaler)

# Save GRU results
pd.DataFrame({"Actual": y_true_inv_gru.flatten(), "Predicted": yhat_inv_gru.flatten()}).to_csv("gru_predictions.csv")
plt.figure(figsize=(10, 4))
plt.plot(y_true_inv_gru, label="Actual")
plt.plot(yhat_inv_gru, label="Predicted")
plt.title("GRU Forecast")
plt.legend()
plt.savefig("gru_forecast.png")
plt.close()

# --- Train and evaluate LSTM ---
print("\n>>> Training LSTM")
lstm_model = create_model_lstm(X_train)
history_lstm = fit_model(lstm_model, X_train, y_train, epochs=50, batch_size=32, validation_data=validation_data)
yhat_lstm = prediction(lstm_model, X_test)
y_true_inv_lstm, yhat_inv_lstm = inverse_transform(y_test, yhat_lstm, scaler)

# Save LSTM results
pd.DataFrame({"Actual": y_true_inv_lstm.flatten(), "Predicted": yhat_inv_lstm.flatten()}).to_csv("lstm_predictions.csv")
plt.figure(figsize=(10, 4))
plt.plot(y_true_inv_lstm, label="Actual")
plt.plot(yhat_inv_lstm, label="Predicted")
plt.title("LSTM Forecast")
plt.legend()
plt.savefig("lstm_forecast.png")
plt.close()

# --- Grid search GRU ---
print("\n>>> Hyperparameter Tuning GRU")
auto_grid_search(create_model_gru, X_train, y_train, X_test, y_test, scaler, validation_data)

# --- Grid search LSTM ---
print("\n>>> Hyperparameter Tuning LSTM")
auto_grid_search(create_model_lstm, X_train, y_train, X_test, y_test, scaler, validation_data)
