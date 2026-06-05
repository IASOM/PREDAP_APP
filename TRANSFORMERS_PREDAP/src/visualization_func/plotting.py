# ==========================================================
# PLOTTING UTILITIES
# A collection of functions for visualizing model predictions,
# error metrics, and other analytical plots.
# ==========================================================

import time
import pandas as pd
import os
import numpy as np
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from src.evaluation.metrics import crps, smape, pinball_loss, mean_absolute_percentage_error

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import ttest_ind, mannwhitneyu
from config.config_manager import get_config

default_config = get_config()

def create_pandemic_waves_df():
    """
    Create a DataFrame with pandemic wave information.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with pandemic wave periods
    """
    # Convert waves dictionary to DataFrame
    df_waves = pd.DataFrame(default_config.PANDEMIC_WAVES).T.reset_index()
    df_waves.columns = ["Onada", "Inici", "Final"]
    df_waves["Inici"] = pd.to_datetime(df_waves["Inici"])
    df_waves["Final"] = pd.to_datetime(df_waves["Final"])
    
    print("Pandemic waves DataFrame created:")
    print(df_waves.to_string(index=False))
    
    return df_waves


def plot_predictions_with_waves(Y_test, predictions, date_list, df_waves, waves=True, plt_show=False, model_name = "Model"):
    """
    Plots actual vs. predicted values with dates as x-labels, showing only 15 evenly spaced date labels.
    Optionally highlights COVID-19 pandemic waves with a red background.

    Args:
        Y_test (array): Actual target values.
        predictions (array): Predicted values from the model.
        date_list (list): List of datetime values for the x-axis.
        df_waves (DataFrame): Contains pandemic waves' start and end dates.
        waves (bool): Whether to highlight pandemic waves. Default is True.
    """
    date_labels = [date.strftime('%Y-%m-%d') for date in date_list]
    num_labels = 15
    indices = np.linspace(0, len(date_list) - 1, num_labels, dtype=int)
    
    plt.figure(figsize=(20, 5))
    
    if waves:
        for _, row in df_waves.iterrows():
            plt.axvspan(row["Inici"], row["Final"], color="red", alpha=0.2)

    plt.plot(date_list, Y_test, marker='o', linestyle='-', alpha=0.7, label="Actual Values")
    plt.plot(date_list, predictions, marker='x', linestyle='--', alpha=0.7, label="Predicted Values")
    
    plt.xlabel("Date")
    plt.ylabel("Target Value (J00)")
    plt.title("Predictions vs. Actual Values (J00 - Y-test)")
    plt.ylim(0, 1)
    plt.xticks([date_list[i] for i in indices], [date_labels[i] for i in indices], rotation=45)
    plt.grid()
    if not os.path.exists("plots/plots_residual_transformer"):
        os.makedirs("plots/plots_residual_transformer", exist_ok=True)
    plt.savefig(f"plots/plots_residual_transformer/predictions_with_waves_{model_name}.png")
    if plt_show:
        plt.show()
    plt.close()
	

def plot_stepwise_errors(Y_test, predictions, model_name = "Model", plt_show = False):
    """
    Computes and plots MSE, MAE, Bias, CRPS, Pinball Loss, and MAPE for each prediction step.

    Args:
        Y_test (array): Actual values (shape: (samples, n_pred)).
        predictions (array): Model predictions (shape: (samples, n_pred)).
    """

    # Ensure inputs are numpy arrays
    Y_test = np.array(Y_test)
    predictions = np.array(predictions)

    # Ensure Y_test has the same shape as predictions
    if Y_test.ndim == 1:
        Y_test = np.tile(Y_test[:, np.newaxis], predictions.shape[1])

    n_steps = predictions.shape[1]  # Number of prediction steps

    mse_steps = []
    mae_steps = []
    bias_steps = []
    crps_steps = []
    pinball_steps = []
    mape_steps = []

    for step in range(n_steps):
        y_true = Y_test[:, step]
        y_pred = predictions[:, step]

        mse_steps.append(mean_squared_error(y_true, y_pred))
        mae_steps.append(mean_absolute_error(y_true, y_pred))
        bias_steps.append(np.mean(y_pred - y_true))
        crps_steps.append(crps(y_true, y_pred))
        pinball_steps.append(pinball_loss(y_true, y_pred))  # τ=0.5
        mape_steps.append(smape(y_true, y_pred))

    # Print average values across all steps
    print(f"🔹 **Average MSE:** {np.mean(mse_steps):.4f}")
    print(f"🔹 **Average MAE:** {np.mean(mae_steps):.4f}")
    print(f"🔹 **Average Bias:** {np.mean(bias_steps):.4f}")
    print(f"🔹 **Average CRPS:** {np.mean(crps_steps):.4f}")
    print(f"🔹 **Average Pinball Loss:** {np.mean(pinball_steps):.4f}")
    print(f"🔹 **Average SMAPE:** {np.mean(mape_steps):.2f}%")

    # Create a range for prediction steps
    step_range = np.arange(1, n_steps + 1)

    # Plot errors
    plt.figure(figsize=(12, 4))
    plt.plot(step_range, mse_steps, label="MSE", marker='o', linestyle='-')
    plt.plot(step_range, mae_steps, label="MAE", marker='x', linestyle='--')
    plt.plot(step_range, bias_steps, label="Bias", marker='s', linestyle='-')
    plt.plot(step_range, crps_steps, label="CRPS", marker='^', linestyle='--')
    plt.plot(step_range, pinball_steps, label="Pinball Loss", marker='d', linestyle=':')
    #plt.plot(step_range, mape_steps, label="MAPE (%)", marker='v', linestyle='-.')

    plt.xlabel("Prediction Step")
    plt.ylabel("Error Value")
    plt.title("Stepwise Errors (MSE, MAE, Bias, CRPS, Pinball Loss, SMAPE)")
    plt.legend()
    plt.grid()
    plt.xticks(step_range)
    if not os.path.exists("plots/plots_residual_transformer"):
        os.makedirs("plots/plots_residual_transformer", exist_ok=True)
    
    plt.savefig(f"plots/plots_residual_transformers/stepwise_errors_{model_name}.png")
    if plt_show:
        plt.show()
    plt.close()
	
def plot_errors_over_time_with_waves(Y_test, predictions, date_list, df_waves, waves=True, ylim=None, model_name = "Model", plt_show=False):
    """
    Plots four separate graphs for MSE & MAE, Bias, CRPS, and Pinball Loss over time.
    Optionally highlights COVID-19 pandemic waves with a red background.

    Args:
        Y_test (array): Actual values (shape: (samples,) or (samples, n_pred)).
        predictions (array): Model predictions (shape: (samples,) or (samples, n_pred)).
        date_list (list): List of datetime values for the x-axis.
        df_waves (DataFrame): Contains pandemic waves' start and end dates.
        waves (bool): If True, highlights pandemic waves on the plots.
        ylim (list, optional): List of y-limits [(ymin, ymax), (ymin, ymax), ...] for each plot.
    """
    # Ensure inputs are numpy arrays
    Y_test = np.array(Y_test)
    predictions = np.array(predictions)

    # Ensure Y_test and predictions have matching dimensions
    if predictions.ndim == 1:
        predictions = predictions[:, np.newaxis]  # Convert to shape (samples, 1)
    if Y_test.ndim == 1:
        Y_test = np.tile(Y_test[:, np.newaxis], predictions.shape[1])  # Expand Y_test

    # Compute errors over time for each timestep
    mse_values = [mean_squared_error(Y_test[i, :], predictions[i, :]) for i in range(len(Y_test))]
    mae_values = [mean_absolute_error(Y_test[i, :], predictions[i, :]) for i in range(len(Y_test))]
    bias_values = [np.mean(predictions[i, :] - Y_test[i, :]) for i in range(len(Y_test))]
    crps_values = [crps(Y_test[i, :], predictions[i, :]) for i in range(len(Y_test))]
    pinball_values = [pinball_loss(Y_test[i, :], predictions[i, :]) for i in range(len(Y_test))]

    # Ensure date_list length matches metric lists
    if len(date_list) != len(mse_values):
        date_list = date_list[-len(mse_values):]  # Trim date_list if necessary

    # Print average values across all timesteps
    print(f"🔹 **Average MSE:** {np.mean(mse_values):.4f}")
    print(f"🔹 **Average MAE:** {np.mean(mae_values):.4f}")
    print(f"🔹 **Average Bias:** {np.mean(bias_values):.4f}")
    print(f"🔹 **Average CRPS:** {np.mean(crps_values):.4f}")
    print(f"🔹 **Average Pinball Loss:** {np.mean(pinball_values):.4f}")

    # Convert timestamps to string format (YYYY-MM-DD)
    date_labels = [date.strftime('%Y-%m-%d') for date in date_list]

    # Select 15 evenly spaced indices for x-axis labels
    num_labels = 15
    indices = np.linspace(0, len(date_list) - 1, num_labels, dtype=int)

    def plot_metric(values, title, ylabel, marker, linestyle, color, ylim_range=None, model_name="Model", plt_show=False):
        plt.figure(figsize=(15, 3))
        if waves:
            for _, row in df_waves.iterrows():
                plt.axvspan(pd.to_datetime(row["Inici"]), pd.to_datetime(row["Final"]), color="red", alpha=0.2)
        plt.plot(date_list, values, label=title, marker=marker, linestyle=linestyle, alpha=0.7, color=color)
        plt.xlabel("Date")
        plt.ylabel(ylabel)
        plt.title(f"{title} Over Time" + (" with Pandemic Waves" if waves else ""))
        plt.legend()
        plt.xticks([date_list[i] for i in indices], [date_labels[i] for i in indices], rotation=45)
        plt.grid()
        
        # Apply y-limit if provided
        if ylim_range is not None:
            plt.ylim(ylim_range)
        if not os.path.exists("plots/plots_residual_transformer"):
            os.makedirs("plots/plots_residual_transformer", exist_ok=True)
        plt.savefig(f"plots/plots_residual_transformers/{title.replace(' ', '_').lower()}_over_time_{model_name}.png")
        if plt_show:
            plt.show()
        plt.close()

    # Default ylim handling: If not provided, use None
    if ylim is None:
        ylim = [None] * 5  # No y-limits applied

    # Plot each metric with optional y-limit
    plot_metric(mse_values, "MSE", "Error Value", 'o', '-', 'blue', ylim[0], model_name=model_name)
    plot_metric(mae_values, "MAE", "Error Value", 'x', '--', 'red', ylim[1], model_name=model_name)
    plot_metric(bias_values, "Bias", "Bias", 's', '-', 'purple', ylim[2], model_name=model_name)
    plot_metric(crps_values, "CRPS", "CRPS", '^', '--', 'green', ylim[3], model_name=model_name)
    plot_metric(pinball_values, "Pinball Loss", "Pinball Loss", 'd', ':', 'orange', ylim[4], model_name=model_name)



def evaluate_error_significance_pandemic_waves(Y_test, predictions, date_list, df_waves):
    """
    Computes error metrics and tests whether errors inside pandemic waves are significantly different 
    from those outside.

    Args:
        Y_test (array): Actual values (shape: (samples, n_pred)).
        predictions (array): Model predictions (shape: (samples, n_pred)).
        date_list (list): Corresponding dates for the errors.
        df_waves (DataFrame): Start and end dates of pandemic waves.

    Returns:
        Prints test results comparing errors inside and outside pandemic waves.
    """

    # Ensure inputs are numpy arrays
    Y_test = np.array(Y_test)
    predictions = np.array(predictions)

    # Ensure Y_test has the same shape as predictions
    if predictions.ndim == 1:
        predictions = predictions[:, np.newaxis]  # Convert to shape (samples, 1)
    if Y_test.ndim == 1:
        Y_test = np.tile(Y_test[:, np.newaxis], predictions.shape[1])  # Expand Y_test

    # Convert dates to pandas DateTime for easy comparison
    date_list = pd.to_datetime(date_list)
    df_waves["Inici"] = pd.to_datetime(df_waves["Inici"])
    df_waves["Final"] = pd.to_datetime(df_waves["Final"])

    # Compute errors dynamically
    mse_values = [mean_squared_error(Y_test[i, :], predictions[i, :]) for i in range(len(Y_test))]
    mae_values = [mean_absolute_error(Y_test[i, :], predictions[i, :]) for i in range(len(Y_test))]
    bias_values = [np.mean(predictions[i, :] - Y_test[i, :]) for i in range(len(Y_test))]
    crps_values = [crps(Y_test[i, :], predictions[i, :]) for i in range(len(Y_test))]
    pinball_values = [pinball_loss(Y_test[i, :], predictions[i, :]) for i in range(len(Y_test))]

    # Separate errors into two groups: inside and outside pandemic waves
    errors_inside_wave = {"MSE": [], "MAE": [], "Bias": [], "CRPS": [], "Pinball Loss": []}
    errors_outside_wave = {"MSE": [], "MAE": [], "Bias": [], "CRPS": [], "Pinball Loss": []}

    for i, date in enumerate(date_list):
        in_wave = any((date >= row["Inici"]) and (date <= row["Final"]) for _, row in df_waves.iterrows())
        
        if in_wave:
            errors_inside_wave["MSE"].append(mse_values[i])
            errors_inside_wave["MAE"].append(mae_values[i])
            errors_inside_wave["Bias"].append(bias_values[i])
            errors_inside_wave["CRPS"].append(crps_values[i])
            errors_inside_wave["Pinball Loss"].append(pinball_values[i])
        else:
            errors_outside_wave["MSE"].append(mse_values[i])
            errors_outside_wave["MAE"].append(mae_values[i])
            errors_outside_wave["Bias"].append(bias_values[i])
            errors_outside_wave["CRPS"].append(crps_values[i])
            errors_outside_wave["Pinball Loss"].append(pinball_values[i])

    # 📌 Perform statistical tests for each error metric
    for metric in errors_inside_wave.keys():
        inside = np.array(errors_inside_wave[metric])
        outside = np.array(errors_outside_wave[metric])

        print(f"\n📊 **Results for {metric}:**")
        print(f"🔹 Inside Pandemic Waves: Mean={inside.mean():.4f}, Std={inside.std():.4f}, N={len(inside)}")
        print(f"🔹 Outside Pandemic Waves: Mean={outside.mean():.4f}, Std={outside.std():.4f}, N={len(outside)}")

        if len(inside) > 1 and len(outside) > 1:  # Ensure we have enough samples for statistical tests
            # Welch's t-test (assumes normal but allows unequal variance)
            t_stat, p_ttest = ttest_ind(inside, outside, equal_var=False)

            # Mann-Whitney U test (non-parametric test for distributions)
            u_stat, p_mannwhitney = mannwhitneyu(inside, outside, alternative="two-sided")

            print(f"✅ **Welch’s t-test:** t={t_stat:.4f}, p={p_ttest:.4f}")
            print(f"✅ **Mann-Whitney U test:** U={u_stat:.4f}, p={p_mannwhitney:.4f}")

            # Interpretation
            alpha = 0.05  # Significance threshold

            if p_ttest < alpha:
                print("🔴 **Significant difference in errors (Welch’s t-test, p < 0.05)**")
            else:
                print("🟢 No significant difference in errors (Welch’s t-test, p >= 0.05)")

            if p_mannwhitney < alpha:
                print("🔴 **Significant difference in errors (Mann-Whitney U test, p < 0.05)**")
            else:
                print("🟢 No significant difference in errors (Mann-Whitney U test, p >= 0.05)")
        else:
            print("⚠️ Not enough samples to perform statistical tests.")
