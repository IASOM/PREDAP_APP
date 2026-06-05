"""
Training and Evaluation Functions for Univariate Transformer
===========================================================
Contains functions for model training, evaluation metrics, and performance analysis.
"""

import tensorflow as tf
import numpy as np
import os
import pickle
from src.utils.mlflow_logger import MLflowLogger
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.core.config_manager import get_config

default_config = get_config()


def train_given_model_and_data(model, X, Y, batch_size=1024, model_name=None, epochs=100, 
                               save_history=False, save_model=True, save_memory=True, 
                               shuffle=False, callbacks=None, patience = 10):
    """
    Train a given model with provided data and save results.
    
    Args:
        model: Compiled Keras model to train
        X: Training input data
        Y: Training target data
        batch_size: Batch size for training
        model_name: Name to save the model
        epochs: Maximum number of training epochs
        save_history: Whether to save training history
        save_model: Whether to save the trained model
        save_memory: Whether to log GPU memory usage
        shuffle: Whether to shuffle training data
        callbacks: List of Keras callbacks
    """
    if save_memory:  # configure GPU memory growth
        # prepare for measuring memory
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
                
    if callbacks is None:  # define callbacks (If no callbacks are provided, it automatically enables Early Stopping)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=patience, restore_best_weights=True)
        callbacks = [early_stop]

    if not os.path.exists(f'{model_name}'):  # check if model exist and run if not
        history = model.fit(x=X, 
                            y=Y, 
                            batch_size=batch_size,  # batch gradient descent (batch size 1024)
                            epochs=epochs, 
                            shuffle=shuffle,        # Allows shuffling
                            validation_split=0.3,   # 30% data for validation
                            callbacks=callbacks)
    else:
        print(f"model {model_name} already exists")
        return

    if model_name is None:
        model_name = "testing"

    if save_history:  # save training history
        raw_model_name = model_name.replace('.keras', '')
        os.makedirs('../history', exist_ok=True)
        clean_history = {
            metric: [float(val) for val in values]
            for metric, values in history.history.items()
        }
        
        with open(f'../history/{raw_model_name}_history.pkl', 'wb') as file_pi:
            pickle.dump(clean_history, file_pi)
            
        # Optional: Delete the cloned dictionary to free up CPU memory immediately
        del clean_history
    
    if save_model and epochs > 1:  # save model
        os.makedirs(default_config.model_folder, exist_ok=True)
        model.save(default_config.model_folder +"/" + model_name)

    if save_memory:  # log memory usage (optional)
        # save memory usage
        # Get memory information
        memory_info = tf.config.experimental.get_memory_info('GPU:0')
        with open('memory.csv', 'a') as resultcsv:
            resultcsv.write(f"{model_name},{memory_info['peak']},train\n")
        print(f"Current memory usage: {memory_info['current'] / (1024**2)} MB")
        print(f"Peak memory usage: {memory_info['peak'] / (1024**2)} MB")


def evaluate_model_sliding_window(model, model_name, X_test, Y_test, date_list, df_waves, sliding_window=10, show_plt=True):
    """
    Evaluate the model using a sliding window approach with pandemic wave visualization.

    Args:
        model: Trained Keras model
        model_name: Name of the model for saving plots
        X_test: Test input data (shape: (samples, lookback, 1))
        Y_test: True future values (shape: (samples, n_pred))
        date_list: List of corresponding dates
        df_waves: DataFrame containing pandemic waves information
        sliding_window: Number of test samples to slide over
    
    Returns:
        Lists of MSE, MAE, and Loss values over time
    """
    import matplotlib.pyplot as plt

    mse_list = []
    mae_list = []
    loss_list = []

    num_samples = len(X_test) - sliding_window + 1  # Number of sliding steps

    for i in range(num_samples):
        X_window = X_test[i : i + sliding_window]  # Get sliding window input
        Y_window_true = Y_test[i : i + sliding_window]  # True values

        # Predict using the model
        Y_window_pred = model.predict(X_window, verbose=0)

        # Compute metrics
        mse = mean_squared_error(Y_window_true, Y_window_pred)
        mae = mean_absolute_error(Y_window_true, Y_window_pred)
        loss = np.mean(np.abs(Y_window_true - Y_window_pred))  # Approximate loss (MAE)

        # Store metrics
        mse_list.append(mse)
        mae_list.append(mae)
        loss_list.append(loss)
    
    # Ensure date_list length matches metric lists
    trimmed_dates = date_list[-len(mse_list):]  # Take only the last elements

    plt.figure(figsize=(12, 10))

    # Plot MSE
    ax1 = plt.subplot(3, 1, 1)
    for i, row in df_waves.iterrows():
        ax1.axvspan(row["Inici"], row["Final"], color="red", alpha=0.2)
    ax1.plot(trimmed_dates, mse_list, label="MSE", color="blue")
    ax1.set_xlabel("Data")
    ax1.set_ylabel("MSE")
    ax1.set_title("Mean Squared Error (MSE) Over Time")
    ax1.legend()
    ax1.tick_params(axis="x", rotation=45)

    # Plot MAE
    ax2 = plt.subplot(3, 1, 2)
    for i, row in df_waves.iterrows():
        ax2.axvspan(row["Inici"], row["Final"], color="red", alpha=0.2)
    ax2.plot(trimmed_dates, mae_list, label="MAE", color="orange")
    ax2.set_xlabel("Data")
    ax2.set_ylabel("MAE")
    ax2.set_title("Mean Absolute Error (MAE) Over Time")
    ax2.legend()
    ax2.tick_params(axis="x", rotation=45)

    # Plot Loss
    ax3 = plt.subplot(3, 1, 3)
    for i, row in df_waves.iterrows():
        ax3.axvspan(row["Inici"], row["Final"], color="red", alpha=0.2)
    ax3.plot(trimmed_dates, loss_list, label="Loss", color="red")
    ax3.set_xlabel("Data")
    ax3.set_ylabel("Loss")
    ax3.set_title("Loss Over Time")
    ax3.legend()
    ax3.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    
    os.makedirs("plots", exist_ok=True)
    out_path = f"plots/evaluation_metrics_over_time_{model_name}.png"
    plt.savefig(out_path)
    MLflowLogger(active=True).log_artifact(out_path, artifact_path="plots")
    if show_plt:
        plt.show()

    return mse_list, mae_list, loss_list


def evaluate_model_basic(model, X_test, Y_test, sliding_window=10, plt_results=True):
    """
    Basic model evaluation using a sliding window approach without date information.

    Args:
        model: Trained Keras model
        X_test: Test input data (shape: (samples, lookback, 1))
        Y_test: True future values (shape: (samples, n_pred))
        sliding_window: Number of test samples to slide over
        plt_results: Whether to plot the results
    
    Returns:
        Lists of MSE, MAE, and Loss values
    """
    mse_list = []
    mae_list = []
    loss_list = []

    num_samples = len(X_test) - sliding_window + 1  # Number of sliding steps

    for i in range(num_samples):
        X_window = X_test[i : i + sliding_window]  # Get sliding window input
        Y_window_true = Y_test[i : i + sliding_window]  # True values

        # Predict using the model
        Y_window_pred = model.predict(X_window, verbose=0)

        # Compute metrics
        mse = mean_squared_error(Y_window_true, Y_window_pred)
        mae = mean_absolute_error(Y_window_true, Y_window_pred)
        loss = np.mean(np.abs(Y_window_true - Y_window_pred))  # Approximate loss (MAE)

        # Store metrics
        mse_list.append(mse)
        mae_list.append(mae)
        loss_list.append(loss)

    # Plot results
    if plt_results:
        plot_evaluations(mse_list, mae_list, loss_list)

    return mse_list, mae_list, loss_list


def plot_evaluations(mse_list, mae_list, loss_list, show_plt=True):
    """
    Plot MSE, MAE, and Loss over time (basic version without dates).

    Args:
        mse_list: List of Mean Squared Error values
        mae_list: List of Mean Absolute Error values
        loss_list: List of Loss values
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 5))

    # Plot MSE
    plt.subplot(3, 1, 1)
    plt.plot(mse_list, label="MSE")
    plt.xlabel("Time")
    plt.ylabel("MSE")
    plt.title("Mean Squared Error (MSE)")
    plt.legend()

    # Plot MAE
    plt.subplot(3, 1, 2)
    plt.plot(mae_list, label="MAE", color="orange")
    plt.xlabel("Time")
    plt.ylabel("MAE")
    plt.title("Mean Absolute Error (MAE)")
    plt.legend()

    # Plot Loss
    plt.subplot(3, 1, 3)
    plt.plot(loss_list, label="Loss", color="red")
    plt.xlabel("Time")
    plt.ylabel("Loss")
    plt.title("Loss Over Time")
    plt.legend()

    plt.tight_layout()
    if show_plt:
        plt.show()