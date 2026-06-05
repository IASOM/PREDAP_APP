import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .metrics import crps, pinball_loss, smape
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import ttest_ind, mannwhitneyu

def plot_predictions_with_waves(Y_test, predictions, date_list, df_waves, waves=True, plt_show=False, model_name = "Model"):
    date_labels = [date.strftime('%Y-%m-%d') for date in date_list]
    num_labels = 15
    indices = np.linspace(0, len(date_list) - 1, num_labels, dtype=int)
    plt.figure(figsize=(20, 5))
    if waves:
        for _, row in df_waves.iterrows():
            plt.axvspan(pd.to_datetime(row["Inici"]), pd.to_datetime(row["Final"]), color="red", alpha=0.2)
    plt.plot(date_list, Y_test, marker='o', linestyle='-', alpha=0.7, label="Actual Values")
    plt.plot(date_list, predictions, marker='x', linestyle='--', alpha=0.7, label="Predicted Values")
    plt.xlabel("Date")
    plt.ylabel("Target Value")
    plt.title(f"Predictions vs. Actual Values ({model_name})")
    plt.grid()
    os.makedirs("plots/plots_residual_transformers", exist_ok=True)
    plt.savefig(f"plots/plots_residual_transformers/predictions_with_waves_{model_name}.png")
    if plt_show:
        plt.show()
    plt.close()

def plot_stepwise_errors(Y_test, predictions, model_name = "Model", plt_show = False):
    Y_test = np.array(Y_test)
    predictions = np.array(predictions)
    if Y_test.ndim == 1:
        Y_test = np.tile(Y_test[:, np.newaxis], predictions.shape[1])
    n_steps = predictions.shape[1]
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
        pinball_steps.append(pinball_loss(y_true, y_pred))
        mape_steps.append(smape(y_true, y_pred))
    print(f"Average MSE: {np.mean(mse_steps):.4f}")
    os.makedirs("plots/plots_residual_transformers", exist_ok=True)
    plt.figure(figsize=(12, 4))
    step_range = np.arange(1, n_steps + 1)
    plt.plot(step_range, mse_steps, label="MSE", marker='o')
    plt.plot(step_range, mae_steps, label="MAE", marker='x')
    plt.plot(step_range, bias_steps, label="Bias", marker='s')
    plt.plot(step_range, crps_steps, label="CRPS", marker='^')
    plt.plot(step_range, pinball_steps, label="Pinball Loss", marker='d')
    plt.legend()
    plt.grid()
    plt.savefig(f"plots/plots_residual_transformers/stepwise_errors_{model_name}.png")
    if plt_show:
        plt.show()
    plt.close()

def plot_errors_over_time_with_waves(Y_test, predictions, date_list, df_waves, waves=True, ylim=None, model_name = "Model", plt_show=False):
    Y_test = np.array(Y_test)
    predictions = np.array(predictions)
    if predictions.ndim == 1:
        predictions = predictions[:, np.newaxis]
    if Y_test.ndim == 1:
        Y_test = np.tile(Y_test[:, np.newaxis], predictions.shape[1])
    mse_values = [mean_squared_error(Y_test[i, :], predictions[i, :]) for i in range(len(Y_test))]
    mae_values = [mean_absolute_error(Y_test[i, :], predictions[i, :]) for i in range(len(Y_test))]
    bias_values = [np.mean(predictions[i, :] - Y_test[i, :]) for i in range(len(Y_test))]
    crps_values = [crps(Y_test[i, :], predictions[i, :]) for i in range(len(Y_test))]
    pinball_values = [pinball_loss(Y_test[i, :], predictions[i, :]) for i in range(len(Y_test))]
    date_labels = [date.strftime('%Y-%m-%d') for date in date_list]
    num_labels = 15
    indices = np.linspace(0, len(date_list) - 1, num_labels, dtype=int)
    def plot_metric(values, title, ylabel, marker, linestyle, color, ylim_range=None):
        plt.figure(figsize=(15, 3))
        if waves:
            for _, row in df_waves.iterrows():
                plt.axvspan(pd.to_datetime(row["Inici"]), pd.to_datetime(row["Final"]), color="red", alpha=0.2)
        plt.plot(date_list, values, label=title, marker=marker, linestyle=linestyle, alpha=0.7, color=color)
        plt.xlabel("Date")
        plt.ylabel(ylabel)
        plt.title(f"{title} Over Time")
        plt.xticks([date_list[i] for i in indices], [date_labels[i] for i in indices], rotation=45)
        plt.grid()
        os.makedirs("plots/plots_residual_transformers", exist_ok=True)
        plt.savefig(f"plots/plots_residual_transformers/{title.replace(' ', '_').lower()}_over_time_{model_name}.png")
        plt.close()
    plot_metric(mse_values, "MSE", "Error Value", 'o', '-', 'blue', None)
    plot_metric(mae_values, "MAE", "Error Value", 'x', '--', 'red', None)
    plot_metric(bias_values, "Bias", "Bias", 's', '-', 'purple', None)
    plot_metric(crps_values, "CRPS", "CRPS", '^', '--', 'green', None)
    plot_metric(pinball_values, "Pinball Loss", "Pinball Loss", 'd', ':', 'orange', None)

def evaluate_error_significance_pandemic_waves(Y_test, predictions, date_list, df_waves):
    Y_test = np.array(Y_test)
    predictions = np.array(predictions)
    if predictions.ndim == 1:
        predictions = predictions[:, np.newaxis]
    if Y_test.ndim == 1:
        Y_test = np.tile(Y_test[:, np.newaxis], predictions.shape[1])
    date_list = pd.to_datetime(date_list)
    df_waves["Inici"] = pd.to_datetime(df_waves["Inici"])
    df_waves["Final"] = pd.to_datetime(df_waves["Final"])
    mse_values = [mean_squared_error(Y_test[i, :], predictions[i, :]) for i in range(len(Y_test))]
    mae_values = [mean_absolute_error(Y_test[i, :], predictions[i, :]) for i in range(len(Y_test))]
    bias_values = [np.mean(predictions[i, :] - Y_test[i, :]) for i in range(len(Y_test))]
    crps_values = [crps(Y_test[i, :], predictions[i, :]) for i in range(len(Y_test))]
    pinball_values = [pinball_loss(Y_test[i, :], predictions[i, :]) for i in range(len(Y_test))]
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
    for metric in errors_inside_wave.keys():
        inside = np.array(errors_inside_wave[metric])
        outside = np.array(errors_outside_wave[metric])
        print(f"Results for {metric}: Inside mean={inside.mean() if len(inside)>0 else float('nan')}, Outside mean={outside.mean() if len(outside)>0 else float('nan')}")
        if len(inside) > 1 and len(outside) > 1:
            t_stat, p_ttest = ttest_ind(inside, outside, equal_var=False)
            u_stat, p_mannwhitney = mannwhitneyu(inside, outside, alternative="two-sided")
            print(f"Welch t-test: p={p_ttest:.4f}, Mann-Whitney: p={p_mannwhitney:.4f}")
