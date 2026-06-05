"""
Visualization Functions for Univariate Transformer
=================================================
Contains plotting functions for model results, predictions, and data exploration.
"""

import matplotlib.pyplot as plt
from src.utils.mlflow_logger import MLflowLogger
import pandas as pd
import numpy as np
import seaborn as sns
import os
from sklearn.metrics import mean_absolute_error
from config.config_manager import get_config
from src.utils.mlflow_logger import MLflowLogger

mlflow_logger = MLflowLogger(active=True)

default_config = get_config()

def plt_model(y_test_inverse, yhat_inverse, date_list, model_name, ci=1.96, show_plt=False):
    """
    Plot model results comparing true vs predicted values across a forecast horizon,
    showing mean and confidence regions computed from multiple prediction samples.

    Parameters:
    -----------
    y_test_inverse : array-like, shape (num_of_predictions, forecast_horizon)
        True values (inverse transformed) for each prediction sample.
    yhat_inverse : array-like, shape (num_of_predictions, forecast_horizon)
        Predicted values (inverse transformed) for each prediction sample.
    model_name : str
        Name of the model for the plot title / filename.
    ci : float
        Multiplier for standard deviation to plot confidence region (default 1.96 ≈ 95%).
    show_plt : bool
        Whether to display the plot interactively.
    """
    try:
        y_test_arr = np.asarray(y_test_inverse)
        yhat_arr = np.asarray(yhat_inverse)

        # Accept 1D inputs by treating them as single-sample predictions
        if y_test_arr.ndim == 1:
            y_test_arr = y_test_arr.reshape(-1, 1)
        if yhat_arr.ndim == 1:
            yhat_arr = yhat_arr.reshape(-1, 1)

        if y_test_arr.shape != yhat_arr.shape:
            raise ValueError("y_test_inverse and yhat_inverse must have the same shape "
                             "(num_of_predictions, forecast_horizon)")

        n_samples, horizon = y_test_arr.shape
        x = np.arange(1, horizon + 1)

        # Compute statistics across prediction samples (axis=0 -> across samples for each horizon step)
        mean_true = np.nanmean(y_test_arr, axis=1)
        std_true = np.nanstd(y_test_arr, axis=1)

        mean_pred = np.nanmean(yhat_arr, axis=1)
        std_pred = np.nanstd(yhat_arr, axis=1)


        # pick middle column (robust to 1D arrays)
        if y_test_arr.ndim > 1:
            middle_value = int(y_test_arr.shape[1] // 2)
        else:
            middle_value = 0
        fig, ax = plt.subplots(figsize=(20, 8))
        # Plot mean lines
        ax.plot(date_list, y_test_arr[:, 3], label='True (mean)', marker='o', linestyle='-', alpha=0.9)
        ax.plot(date_list, yhat_arr[:, 3], label='Predicted (mean)', marker='x', linestyle='--', alpha=0.9)

        # Plot confidence regions: mean ± ci * std
        upper_true = mean_true + ci * std_true
        lower_true = mean_true - ci * std_true
        upper_pred = mean_pred + ci * std_pred
        lower_pred = mean_pred - ci * std_pred

        '''ax.fill_between(date_list, lower_true, upper_true, color='blue', alpha=0.15, label=f'True ± {ci}σ')
        ax.fill_between(date_list, lower_pred, upper_pred, color='orange', alpha=0.15, label=f'Predicted ± {ci}σ')
        '''
        ax.set_xlabel('Forecast Horizon Step', fontweight='bold', fontsize=12)
        ax.set_ylabel('Value', fontweight='bold', fontsize=12)
        ax.set_title(f'Real vs. Predicted (mean ± {ci}σ) // MODEL: {model_name}')
        ax.legend()
        ax.grid(alpha=0.3)

        fig.tight_layout()
        os.makedirs("plots", exist_ok=True)
        fig.savefig(f"plots/model_results_{model_name}_horizon.png")
        MLflowLogger(active=True).log_artifact(f"plots/model_results_{model_name}_horizon.png", artifact_path="plots")
        if show_plt:
            plt.show()
        plt.close(fig)

    except Exception as e:
        print(f"Error plotting model results: {str(e)}")


def plot_predictions_with_waves(Y_test, predictions, date_list, df_waves, model_name, show_plt=False):
    """
    Plot actual vs. predicted values with dates as x-labels, showing only 15 evenly spaced date labels.
    Also highlights COVID-19 pandemic waves with a red background.

    Args:
        Y_test (array): Actual target values
        predictions (array): Predicted values from the model
        date_list (list): List of datetime values for the x-axis
        df_waves (DataFrame): Contains pandemic waves' start and end dates
    """
    # Convert timestamps to string format (YYYY-MM-DD)
    date_labels = [date.strftime('%Y-%m-%d') for date in date_list]

    # Select 15 evenly spaced indices for x-axis labels
    num_labels = 15
    indices = np.linspace(0, len(date_list) - 1, num_labels, dtype=int)

    plt.figure(figsize=(20, 5))

    # Highlight pandemic waves with a red background
    for i, row in df_waves.iterrows():
        plt.axvspan(row["Inici"], row["Final"], color="red", alpha=0.2)

    # pick middle column (robust to 1D arrays)
    if Y_test.ndim > 1:
        middle_value = int(Y_test.shape[1] // 2)
    else:
        middle_value = 0
    # Plot actual and predicted values
    plt.plot(date_list, Y_test, label="Actual Values (Y-test)", marker='o', linestyle='-', alpha=0.7)
    plt.plot(date_list, predictions, label="Predicted Values", marker='x', linestyle='--', alpha=0.7)

    plt.xlabel("Date")
    plt.ylabel("Target Value (J00)")
    plt.title("Predictions vs. Actual Values (J00 - Y-test) with Pandemic Waves")
    plt.legend()

    # Apply only 15 labels to the x-axis
    plt.xticks([date_list[i] for i in indices], [date_labels[i] for i in indices], rotation=45)

    plt.grid()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/predictions_with_waves_{model_name}.png")
    MLflowLogger(active=True).log_artifact(f"plots/predictions_with_waves_{model_name}.png", artifact_path="plots")
    if show_plt:
        plt.show()
    plt.close()


def plot_residuals_analysis(original_predictions, corrected_predictions, actual_values, title_prefix="", show_plt=False, model_name = "Model", timestamp=None):
    """
    Plot analysis comparing original predictions, corrected predictions, and actual values.
    Assumes inputs are arrays of shape (num_of_predictions, forecast_horizon).
    This function plots only the middle-horizon forecast as the main line and
    shows uncertainty bands around it computed from the spread across horizons.
    """
    # Ensure numpy arrays
    original_predictions = np.asarray(original_predictions)
    corrected_predictions = np.asarray(corrected_predictions)
    actual_values = np.asarray(actual_values)

    # Determine mid horizon index if 2D, otherwise treat as 1D series
    def _extract_mid_and_spread(arr):
        if arr.ndim == 1:
            mid = arr
            p10 = arr
            p90 = arr
            mn = arr
            mx = arr
        else:
            H = arr.shape[1]
            mid_idx = H // 2
            mid = arr[:, mid_idx]
            # compute different measures of spread across horizons
            p10 = np.percentile(arr, 10, axis=1)
            p90 = np.percentile(arr, 90, axis=1)
            mn = arr.min(axis=1)
            mx = arr.max(axis=1)
        return mid, p10, p90, mn, mx

    orig_mid, orig_p10, orig_p90, orig_min, orig_max = _extract_mid_and_spread(original_predictions)
    corr_mid, corr_p10, corr_p90, corr_min, corr_max = _extract_mid_and_spread(corrected_predictions)
    act_mid, _, _, act_min, act_max = _extract_mid_and_spread(actual_values)

    # Basic length check
    n = len(act_mid)
    if not (len(orig_mid) == n and len(corr_mid) == n):
        raise ValueError("Input arrays must have the same number of prediction timesteps (first dimension).")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{title_prefix} Analysis (middle horizon + across-horizon uncertainty)', fontsize=14)

    # Time series: middle-horizon predictions with uncertainty bands from across horizons
    if timestamp is not None:
        t = pd.to_datetime(timestamp)
    else:
        t = np.arange(n)
    axes[0, 0].plot(t, act_mid, label='Actual (middle horizon)', color='black', alpha=0.9)
    axes[0, 0].plot(t, orig_mid, label='Original (middle)', color='C0', alpha=0.9)
    axes[0, 0].plot(t, corr_mid, label='Corrected (middle)', color='C1', alpha=0.9)

    # Fill uncertainty bands: (min,max) light band and (10th,90th) darker band
    axes[0, 0].fill_between(t, orig_min, orig_max, color='C0', alpha=0.22, label='Original min-max')
    axes[0, 0].fill_between(t, orig_p10, orig_p90, color='C0', alpha=0.35, label='Original 10-90 pct')

    axes[0, 0].fill_between(t, corr_min, corr_max, color='C1', alpha=0.22, label='Corrected min-max')
    axes[0, 0].fill_between(t, corr_p10, corr_p90, color='C1', alpha=0.35, label='Corrected 10-90 pct')

    axes[0, 0].set_title('Middle-horizon Time Series with Across-horizon Uncertainty')
    axes[0, 0].set_xlabel('Time Steps')
    axes[0, 0].set_ylabel('Values')
    axes[0, 0].legend(loc='upper left', fontsize='small')
    axes[0, 0].grid(True, alpha=0.3)

    # Residuals for middle horizon and a shaded band for residual spread across horizons
    orig_res_mid = act_mid - orig_mid
    corr_res_mid = act_mid - corr_mid

    # For residual spread compute residuals across horizons then percentiles
    if original_predictions.ndim == 2:
        orig_res_all = actual_values - original_predictions
        orig_res_p10 = np.percentile(orig_res_all, 10, axis=1)
        orig_res_p90 = np.percentile(orig_res_all, 90, axis=1)
    else:
        orig_res_p10 = orig_res_p90 = orig_res_mid

    if corrected_predictions.ndim == 2:
        corr_res_all = actual_values - corrected_predictions
        corr_res_p10 = np.percentile(corr_res_all, 10, axis=1)
        corr_res_p90 = np.percentile(corr_res_all, 90, axis=1)
    else:
        corr_res_p10 = corr_res_p90 = corr_res_mid

    axes[0, 1].plot(t, orig_res_mid, label='Original residuals (middle)', color='C0', alpha=0.9)
    axes[0, 1].fill_between(t, orig_res_p10, orig_res_p90, color='C0', alpha=0.2, label='Original resid 10-90 pct')

    axes[0, 1].plot(t, corr_res_mid, label='Corrected residuals (middle)', color='C1', alpha=0.9)
    axes[0, 1].fill_between(t, corr_res_p10, corr_res_p90, color='C1', alpha=0.2, label='Corrected resid 10-90 pct')

    axes[0, 1].axhline(0, color='black', linestyle='--', alpha=0.6)
    axes[0, 1].set_title('Residuals (middle horizon) with Residual Spread')
    axes[0, 1].set_xlabel('Time Steps')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].legend(fontsize='small')
    axes[0, 1].grid(True, alpha=0.3)

    # Scatter: actual vs predicted (middle horizon)
    axes[1, 0].scatter(act_mid, orig_mid, alpha=0.6, label='Original (middle)', color='C0')
    axes[1, 0].scatter(act_mid, corr_mid, alpha=0.6, label='Corrected (middle)', color='C1')

    min_val = min(np.nanmin(act_mid), np.nanmin(orig_min), np.nanmin(corr_min))
    max_val = max(np.nanmax(act_mid), np.nanmax(orig_max), np.nanmax(corr_max))
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')

    axes[1, 0].set_title('Actual vs Predicted (middle horizon)')
    axes[1, 0].set_xlabel('Actual (middle)')
    axes[1, 0].set_ylabel('Predicted (middle)')
    axes[1, 0].legend(fontsize='small')
    axes[1, 0].grid(True, alpha=0.3)

    # Error distribution for middle horizon (absolute)
    orig_errors_mid = np.abs(orig_res_mid)
    corr_errors_mid = np.abs(corr_res_mid)
    axes[1, 1].hist(orig_errors_mid, bins=30, alpha=0.6, label=f'Original (MAE: {mean_absolute_error(actual_values, original_predictions):.4f})', color='C0')
    axes[1, 1].hist(corr_errors_mid, bins=30, alpha=0.6, label=f'Corrected (MAE: {mean_absolute_error(actual_values, corrected_predictions):.4f})', color='C1')
    axes[1, 1].set_title('Error Distribution (middle horizon)')
    axes[1, 1].set_xlabel('Absolute Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend(fontsize='small')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(default_config.plots_dir, exist_ok=True)
    out_path = f"{default_config.plots_dir}/residuals_analysis_{model_name}.png"
    plt.savefig(out_path)
    mlflow_logger.log_artifact(out_path, artifact_path="plots")
    if show_plt:
        plt.show()
    plt.close()
