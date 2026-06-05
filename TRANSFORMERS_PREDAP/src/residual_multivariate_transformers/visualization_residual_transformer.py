"""
Visualization Module for Residual Multivariate Transformers
===========================================================

This module contains visualization and plotting functions for residual multivariate
transformer models, including prediction plots, error analysis, and pandemic wave visualization.
"""

import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add the src directory to path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir)) if 'residual_multivariate_transformers' in current_dir else os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    import evaluation_plot_utils
except ImportError:
    print("Warning: evaluation_plot_utils module not found. Some plotting functions may not work.")
    evaluation_plot_utils = None

from src.core.config_manager import get_config
from src.utils.mlflow_logger import MLflowLogger

# MLflow logger wrapper (no-op if mlflow is not installed)
mlflow_logger = MLflowLogger(active=True)

default_config = get_config()


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


def plot_stepwise_errors_comparison(Y_test, original_predictions, corrected_predictions, title_prefix="", model_name = "Model"):
    """
    Plot stepwise errors for original and corrected predictions.
    
    Parameters:
    -----------
    Y_test : np.ndarray
        Test target values
    original_predictions : np.ndarray
        Original model predictions
    corrected_predictions : np.ndarray
        Residual-corrected predictions
    title_prefix : str, optional
        Prefix for plot titles
    """
    if evaluation_plot_utils is None:
        print("Warning: evaluation_plot_utils not available. Cannot plot stepwise errors.")
        return
    
    print(f"{title_prefix} - Original Predictions Stepwise Errors:")
    evaluation_plot_utils.plot_stepwise_errors(Y_test, original_predictions, model_name = model_name)
    
    print(f"{title_prefix} - Corrected Predictions Stepwise Errors:")
    evaluation_plot_utils.plot_stepwise_errors(Y_test, corrected_predictions, model_name = model_name)
    
def plot_predictions_with_pandemic_waves(Y_test, predictions, date_list, df_waves=None, title="Predictions with Pandemic Waves", model_name = "Model"):
    """
    Plot predictions overlaid with pandemic wave periods.
    
    Parameters:
    -----------
    Y_test : np.ndarray
        Test target values
    predictions : np.ndarray
        Model predictions
    date_list : list
        List of dates corresponding to predictions
    df_waves : pd.DataFrame, optional
        DataFrame with pandemic wave information
    title : str, optional
        Plot title
    """
    if evaluation_plot_utils is None:
        print("Warning: evaluation_plot_utils not available. Cannot plot with pandemic waves.")
        return
    
    # Create waves DataFrame if not provided
    if df_waves is None:
        df_waves = create_pandemic_waves_df()
    
    # Ensure predictions are 1D for plotting
    if len(predictions.shape) > 1:
        predictions_to_plot = predictions.mean(axis=1)
    else:
        predictions_to_plot = predictions
        
    if len(Y_test.shape) > 1:
        Y_test_to_plot = Y_test.mean(axis=1)
    else:
        Y_test_to_plot = Y_test
    
    print(f"Plotting {title}")
    evaluation_plot_utils.plot_predictions_with_waves(
        Y_test_to_plot, predictions_to_plot, date_list, df_waves, waves=False
    )


def plot_errors_over_time_with_waves(Y_test, predictions, date_list, df_waves=None):
    """
    Plot prediction errors over time with pandemic wave periods.
    
    Parameters:
    -----------
    Y_test : np.ndarray
        Test target values
    predictions : np.ndarray
        Model predictions
    date_list : list
        List of dates corresponding to predictions
    df_waves : pd.DataFrame, optional
        DataFrame with pandemic wave information
    """
    if evaluation_plot_utils is None:
        print("Warning: evaluation_plot_utils not available. Cannot plot errors over time.")
        return
    
    # Create waves DataFrame if not provided
    if df_waves is None:
        df_waves = create_pandemic_waves_df()
    
    # Ensure predictions are 1D for plotting
    if len(predictions.shape) > 1:
        predictions_to_plot = predictions.mean(axis=1)
    else:
        predictions_to_plot = predictions
        
    if len(Y_test.shape) > 1:
        Y_test_to_plot = Y_test.mean(axis=1)
    else:
        Y_test_to_plot = Y_test
    
    print("Plotting errors over time with pandemic waves")
    evaluation_plot_utils.plot_errors_over_time_with_waves(
        Y_test_to_plot, predictions_to_plot, date_list, df_waves, waves=False
    )


def evaluate_error_significance_pandemic_waves(Y_test, predictions, date_list, df_waves=None, model_name = "Model"):
    """
    Evaluate error significance during pandemic waves.
    
    Parameters:
    -----------
    Y_test : np.ndarray
        Test target values
    predictions : np.ndarray
        Model predictions
    date_list : list
        List of dates corresponding to predictions
    df_waves : pd.DataFrame, optional
        DataFrame with pandemic wave information
    """
    if evaluation_plot_utils is None:
        print("Warning: evaluation_plot_utils not available. Cannot evaluate error significance.")
        return
    
    # Create waves DataFrame if not provided
    if df_waves is None:
        df_waves = create_pandemic_waves_df()
    
    # Ensure predictions are 1D for analysis
    if len(predictions.shape) > 1:
        predictions_to_plot = predictions.mean(axis=1)
    else:
        predictions_to_plot = predictions
        
    if len(Y_test.shape) > 1:
        Y_test_to_plot = Y_test.mean(axis=1)
    else:
        Y_test_to_plot = Y_test
    
    print("Evaluating error significance during pandemic waves")
    evaluation_plot_utils.evaluate_error_significance_pandemic_waves(
        Y_test_to_plot, predictions_to_plot, date_list, df_waves
    )


def create_pandemic_waves_df():
    """
    Create a DataFrame with pandemic wave information from config.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with pandemic wave periods
    """
    df_waves = pd.DataFrame(default_config.PANDEMIC_WAVES).T.reset_index()
    df_waves.columns = ["Onada", "Inici", "Final"]
    df_waves["Inici"] = pd.to_datetime(df_waves["Inici"])
    df_waves["Final"] = pd.to_datetime(df_waves["Final"])
    
    return df_waves


def plot_training_history(history, model_name="Model", show_plt=False):
    """
    Plot training history (loss and metrics over epochs).
    
    Parameters:
    -----------
    history : tf.keras.callbacks.History or dict
        Training history object or history dictionary
    model_name : str, optional
        Model name for plot title
    """
    if hasattr(history, 'history'):
        hist_dict = history.history
    else:
        hist_dict = history
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'{model_name} Training History', fontsize=14)
    
    # Plot training loss
    if 'loss' in hist_dict:
        axes[0].plot(hist_dict['loss'], label='Training Loss')
    if 'val_loss' in hist_dict:
        axes[0].plot(hist_dict['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot metrics (MAE if available)
    if 'mae' in hist_dict:
        axes[1].plot(hist_dict['mae'], label='Training MAE')
    if 'val_mae' in hist_dict:
        axes[1].plot(hist_dict['val_mae'], label='Validation MAE')
    elif 'mse' in hist_dict:
        axes[1].plot(hist_dict['mse'], label='Training MSE')
        if 'val_mse' in hist_dict:
            axes[1].plot(hist_dict['val_mse'], label='Validation MSE')
    
    axes[1].set_title('Model Metrics')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Metric Value')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(f"{default_config.plots_dir}/plots_residual_transformers", exist_ok=True)
    out_path = f"{default_config.plots_dir}/plots_residual_transformers/training_history_{model_name}.png"
    plt.savefig(out_path)
    mlflow_logger.log_artifact(out_path, artifact_path="plots")
    if show_plt:
        plt.show()
    plt.close()


def plot_model_comparison(models_results, metric='mae', title="Model Comparison", model_name = "Model", plt_show=False):
    """
    Plot comparison of multiple models' performance.
    
    Parameters:
    -----------
    models_results : dict
        Dictionary with model names as keys and results as values
    metric : str, optional
        Metric to compare ('mae', 'mse', 'loss')
    title : str, optional
        Plot title
    """
    model_names = list(models_results.keys())
    values = [models_results[name][metric] for name in model_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, values, alpha=0.7)
    plt.title(f'{title} - {metric.upper()}')
    plt.xlabel('Model')
    plt.ylabel(metric.upper())
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01, 
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    os.makedirs(f"{default_config.plots_dir}/plots_residual_transformers", exist_ok=True)
    out_path = f"{default_config.plots_dir}/plots_residual_transformers/model_comparison_{metric}_{model_name}.png"
    plt.savefig(out_path)
    mlflow_logger.log_artifact(out_path, artifact_path="plots")
    if plt_show:
        plt.show()
    plt.close()