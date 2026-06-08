# main_Gcausal.py
# ------------------------------------------------------
# Main script for conducting Granger causality analysis on time series data
# Author: Guillem Hernández Guillamet
# Version: 1.0
# Date: 04/06/2025
# Description:
#   Loads and preprocesses time series data, tests for stationarity,
#   applies differencing where needed, selects optimal VAR model lag,
#   fits the VAR model, and generates a Granger causality matrix.
#   Visualization and exploratory functions assist in interpreting results.
# ------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from utils_Gcausal import (
    smoother, stationate, lag_plots,
    kpss_test, splitter, select_p, granger_causation_matrix
)

# --- Load and smooth data ---
df = smart_read('synthetic_timeseries.csv', index_col=0)
df.index = pd.date_range(start="2010-01-01", periods=len(df), freq="D")
df = df.clip(lower=0)

window_size = 14
smoothed = smoother(df, window_size)
smoothed_scaled = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())

# --- Define target and predictors ---
target = ["timeseries_350"]
predictors = ['timeseries_405','timeseries_363','timeseries_975','timeseries_120','timeseries_2',
              'timeseries_541','timeseries_775','timeseries_181','timeseries_692','timeseries_443',
              'timeseries_763','timeseries_31','timeseries_324']
plot_cols = target + predictors

# --- Plot raw series ---
fig, ax = plt.subplots(len(plot_cols), figsize=(20, 15), sharex=True)
df[plot_cols].plot(subplots=True, legend=False, ax=ax)
for i, a in enumerate(ax):
    a.set_ylabel(plot_cols[i])
ax[-1].set_xlabel('')
plt.tight_layout()
plt.show()

# --- Check and ensure stationarity ---
print("\n[INFO] KPSS Test (Initial)...")
display(kpss_test(df[plot_cols]))

# --- First differencing ---
indexes = kpss_test(df[plot_cols]).T[kpss_test(df[plot_cols]).T['p-value'] < 0.05].index.tolist()
stationate_df = stationate(df[plot_cols], indexes)
lag_plots(stationate_df)

# --- Second differencing if needed ---
print("\n[INFO] KPSS Test (Post-1st-diff)...")
display(kpss_test(stationate_df))
indexes = kpss_test(stationate_df).T[kpss_test(stationate_df).T['p-value'] < 0.05].index.tolist()
if indexes:
    stationate_df = stationate(stationate_df, indexes)
    print("[INFO] Applied second differencing.")
    lag_plots(stationate_df)

# --- VAR Training ---
train_df, test_df = splitter(stationate_df)
select_p(train_df)

# --- Fit VAR with selected lag (default p=7 for speed) ---
p = 7
model = VAR(train_df)
var_model = model.fit(p)

# --- Granger causality matrix ---
print("\n[INFO] Granger Causality Matrix (lag=30)...")
display(granger_causation_matrix(train_df, train_df.columns, p=30))
