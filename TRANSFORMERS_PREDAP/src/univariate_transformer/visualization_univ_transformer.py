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


def plot_example(df, title, plt_show=False):
    """
    Plot example of 10 diagnoses from raw data.
    
    Args:
        df: DataFrame with diagnostic data
        title: Title for the plot
    """
    # PLOT RAW DATA (example 10 diags)
    dff = df.copy()
    dff["date"] = dff.index
    dff = dff[['J00', 'COV-19', 'date']]
    sns.set_theme(rc={'figure.figsize': (20, 8)})
    sns.lineplot(data=dff.replace('nan', float('nan')).melt(id_vars=['date']), 
                 x='date', y='value', hue='variable').set(title=title)
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/raw_data_example.png")
    MLflowLogger(active=True).log_artifact("plots/raw_data_example.png", artifact_path="plots")
    if plt_show:
        plt.show()
    plt.close()

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


def plot_training_history(history, model_name, save_plot=True, show_plt=False):
    """
    Plot training history including loss and metrics.
    
    Args:
        history: Keras training history object
        model_name: Name of the model for plot title and saving
        save_plot: Whether to save the plot to file
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training & validation loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Plot training & validation MAE
    if 'mae' in history.history:
        axes[0, 1].plot(history.history['mae'], label='Training MAE')
        if 'val_mae' in history.history:
            axes[0, 1].plot(history.history['val_mae'], label='Validation MAE')
        axes[0, 1].set_title('Model MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
    
    # Plot training & validation MSE
    if 'mse' in history.history:
        axes[1, 0].plot(history.history['mse'], label='Training MSE')
        if 'val_mse' in history.history:
            axes[1, 0].plot(history.history['val_mse'], label='Validation MSE')
        axes[1, 0].set_title('Model MSE')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MSE')
        axes[1, 0].legend()
    
    # Plot learning rate if available
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'], label='Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_plot:
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/training_history_{model_name}.png")
        MLflowLogger(active=True).log_artifact(f"plots/training_history_{model_name}.png", artifact_path="plots")
    if show_plt:
        plt.show()
    plt.close()


def plot_model_comparison(models_results, metric='mae', show_plt=False):
    """
    Compare multiple models' performance.
    
    Args:
        models_results: Dictionary with model names as keys and results as values
        metric: Metric to compare ('mae', 'mse', 'loss')
    """
    model_names = list(models_results.keys())
    values = [results[metric] for results in models_results.values()]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, values)
    plt.title(f'Model Comparison - {metric.upper()}')
    plt.xlabel('Models')
    plt.ylabel(metric.upper())
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/model_comparison_{metric}.png")
    MLflowLogger(active=True).log_artifact(f"plots/model_comparison_{metric}.png", artifact_path="plots")
    if show_plt:
        plt.show()
    plt.close()


def plot_residuals_analysis(y_true, y_pred, model_name, show_plt=False):
    """
    Plot residuals analysis including residual distribution and Q-Q plot.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Residuals vs Predicted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Predicted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Predicted')
    
    # Histogram of residuals
    axes[0, 1].hist(residuals, bins=30, alpha=0.7)
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Histogram of Residuals')
    
    # Time series of residuals
    axes[1, 0].plot(residuals)
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Residuals Over Time')
    
    # Q-Q plot (simplified)
    from scipy import stats
    stats.probplot(residuals.flatten(), dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot')
    
    plt.suptitle(f'Residuals Analysis - {model_name}')
    plt.tight_layout()
    
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/residuals_analysis_{model_name}.png")
    MLflowLogger(active=True).log_artifact(f"plots/residuals_analysis_{model_name}.png", artifact_path="plots")
    if show_plt:
        plt.show()
    plt.close()