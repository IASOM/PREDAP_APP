# main_LMLR.py
# ------------------------------------------------------
# Main script for feature selection in lagged multiple linear regression model
# Author: Guillem Hernández Guillamet
# Version: 1.0
# Date: 04/06/2025
# ------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.utils_LMLR import (
    smoother, plot_example, ploter,
    get_top_correlations_blog, compute_vif, filter_VIF,
    models_training, metrics_calculation
)

# --- Load and preprocess data ---
print("[INFO] Loading data...")
df = smart_read('CCLR_PREDAP/data/synthetic_timeseries.csv', index_col=0)
df.index = pd.date_range(start="2010-01-01", periods=len(df), freq="D")
df = df.clip(lower=0)

plot_example(df, "Raw data (10 examples)")

# --- Smooth and normalize ---
print("[INFO] Smoothing and scaling data...")
window_size = 14
smoothed = df#smoother(df, window_size)
smoothed_scaled = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())
smoothed = smoothed_scaled
plot_example(smoothed, "Smoothed + Scaled Data (10 examples)")

# --- Correlation and VIF filtering ---
code = 'timeseries_1'
correlation_threshold = 0.90
VIF_threshold = 20.0
iterations_max = 400

print("[INFO] Correlation filtering...")
dataframe = smoothed.copy()
df_correlations = get_top_correlations_blog(dataframe, threshold=correlation_threshold)
correlated_vars = list(set(df_correlations.index.get_level_values(0)) | set(df_correlations.index.get_level_values(1)))
non_correlated_vars = list(set(dataframe.columns) - set(correlated_vars))

print("[INFO] VIF filtering...")
vif = compute_vif(correlated_vars, dataframe).sort_values('VIF', ascending=False)
redundant_vars = filter_VIF(vif, dataframe, iterations_max, VIF_threshold)

# --- Final variable selection ---
interesting_vars = redundant_vars + non_correlated_vars
dataframe = dataframe[interesting_vars]
dataframe[code] = smoothed[code]  # Add target back
non_scaled_df = dataframe.copy()
dataframe = (dataframe - dataframe.min()) / (dataframe.max() - dataframe.min())

# Visualize selected variables
ploter(non_scaled_df, "Filtered non-scaled variables", 20, code)
ploter(dataframe, "Filtered scaled variables", 20, code)

# --- Train models ---
print("[INFO] Training baseline model...")
max_iters = 601
smot_corr = non_scaled_df.corrwith(smoothed[code]).sort_values(ascending=False, key=abs)
models, predictors, Y_PRED_train, Y_PRED_test, y_train, y_test = models_training(dataframe, code, smot_corr, max_iters)

print("[INFO] Calculating metrics...")
df_metrics = metrics_calculation(models, y_train, y_test, Y_PRED_train, Y_PRED_test, predictors)

# --- Plot performance ---
sns.lineplot(data=df_metrics[['number_of_variables','MAPE_train','MAPE_test']].melt(id_vars='number_of_variables'),
             x='number_of_variables', y='value', hue='variable').set(title="MAPE vs. Number of Variables")
plt.show()

sns.lineplot(data=df_metrics[['number_of_variables','RMSE_train','RMSE_test']].melt(id_vars='number_of_variables'),
             x='number_of_variables', y='value', hue='variable').set(title="RMSE vs. Number of Variables")
plt.show()

# --- Highlight best model ---
best_model_index = df_metrics['MAPE_test'].idxmin()
print("[INFO] Best model summary:")
print(models[best_model_index].summary())

models[best_model_index].to_excel("BEST_features_NOSMOOTH.xlsx", index=False, engine='openpyxl')