"""
synthetic_data.py

This script loads a time series dataset (e.g., ICD-10 code counts over time),
computes statistical parameters per series, and outputs a CSV file summarizing
key features such as trend, seasonality, sparsity, and noise. The summary is 
useful for generating synthetic time series with realistic characteristics.

Author: Guillem Hernández Guillamet
Date: 2025-06-03
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression

# Load your dataset here (ensure the file exists in the same directory)
# For example:
# df = pd.read_csv("your_real_timeseries.csv", index_col=0, parse_dates=True)
# Uncomment and modify the above line to point to your real data source

# Example placeholder (delete this after adding your own data)
# df = pd.DataFrame()  # This must be replaced

# Check that 'df' exists
try:
    df
except NameError:
    raise ValueError("Dataset 'df' not defined. Please load your DataFrame before running.")

# Initialize list to hold parameter summaries
summary = []

# Time variable for trend analysis
X = np.arange(len(df)).reshape(-1, 1)

# Loop through each ICD code (column)
for col in df.columns:
    series = df[col].fillna(0).values  # Fill NaNs with 0

    # --- Basic statistics ---
    mean_val = np.mean(series)
    std_val = np.std(series)
    sparsity_val = np.sum(series == 0) / len(series)

    # --- Linear trend estimation ---
    model = LinearRegression().fit(X, series)
    trend_slope = model.coef_[0]
    detrended = series - model.predict(X)

    # --- Residual noise statistics ---
    residual_std = np.std(detrended)
    skewness = skew(detrended)
    kurt_val = kurtosis(detrended)

    # --- Seasonality detection via autocorrelation ---
    autocorr_vals = acf(series, nlags=365, fft=True)
    seasonality_lags = np.argsort(autocorr_vals[1:])[::-1][:3] + 1  # exclude lag 0

    # --- Record in summary ---
    summary.append({
        "Code": col,
        "Mean": mean_val,
        "Std": std_val,
        "Trend_Slope": trend_slope,
        "Noise_Std": residual_std,
        "Skewness": skewness,
        "Kurtosis": kurt_val,
        "Sparsity": sparsity_val,
        "Seasonality_Lags": seasonality_lags.tolist()
    })

# Convert to DataFrame
summary_df = pd.DataFrame(summary)

# Save to CSV
summary_df.to_csv("synthetic_timeseries.csv", index=False)
print("Summary saved to synthetic_timeseries.csv")


# --- Synthetic Time Series Generation Section ---

# Parameters
n_series = 1000
series_length = 365 * 10  # 10 years of daily data
rng = np.random.default_rng(seed=42)

# Storage for generated series
synthetic_data = {}

# Generate synthetic series based on sampled summary parameters
for i in range(n_series):
    sample = summary_df.sample(1, random_state=rng.integers(0, 1e6)).iloc[0]

    mean = sample["Mean"]
    std = sample["Std"]
    trend_slope = sample["Trend_Slope"]
    noise_std = sample["Noise_Std"]
    sparsity = sample["Sparsity"]
    lags = sample["Seasonality_Lags"]
    if isinstance(lags, str):
        lags = eval(lags)  # ensure lags are converted to list from string

    # Base time axis
    t = np.arange(series_length)

    # Build seasonal signal
    seasonal = sum(np.sin(2 * np.pi * t / lag) for lag in lags if lag > 0)
    seasonal = (seasonal - np.mean(seasonal)) / np.std(seasonal) * std + mean

    # Add trend
    trend = trend_slope * t

    # Add Gaussian noise
    noise = rng.normal(0, noise_std, size=series_length)

    # Combine components
    ts = seasonal + trend + noise

    # Apply sparsity by zeroing out values
    mask = rng.uniform(0, 1, size=series_length) < sparsity
    ts[mask] = 0

    # Store result
    synthetic_data[f"timeseries_{i+1}"] = ts

# Create and save DataFrame
synthetic_df = pd.DataFrame(synthetic_data)
synthetic_df.index.name = "Day"
synthetic_df.to_csv("synthetic_timeseries.csv")
print("Synthetic time series saved to synthetic_timeseries.csv")
