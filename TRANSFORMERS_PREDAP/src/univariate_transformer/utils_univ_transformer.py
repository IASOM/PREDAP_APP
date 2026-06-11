"""
Utility Functions for Univariate Transformer
===========================================
Contains helper functions for data processing, model management, and file operations.
"""

import re
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from utils.experiments_utils import smart_read

import pickle
import matplotlib.pyplot as plt

from src.core.config_manager import get_config
default_config = get_config()
PLOTS_DIR = default_config.plots_dir

def extract_model_params(model_name):
    """
    Extract lookback and forecast parameters from model filename.
    
    Expected format: {code}_example_transformer_{forecast}fh_{ff_dim}ff_{lookback}lb_{lr}initlr.keras
    
    Args:
        model_name (str): Model filename
        
    Returns:
        tuple: (lookback, forecast) or (None, None) if not found
    """
    # Pattern to match the model name format
    pattern = r'(\d+)fh_\d+ff_(\d+)lb_'
    
    match = re.search(pattern, model_name)
    if match:
        forecast = int(match.group(1))  # First group is forecast
        lookback = int(match.group(2))  # Second group is lookback
        return lookback, forecast
    else:
        print(f"Could not extract parameters from: {model_name}")
        return None, None


from src.core.data_utils import split_train_test


def load_and_evaluate_models(model_folder='models', input_directory=None, code="T14", scaler=None ):
    """
    Load all models in a folder and evaluate them with their respective parameters.
    
    Args:
        model_folder: Path to folder containing .keras models
        input_directory: Path to data file
        code: Diagnostic code to evaluate
        
    Returns:
        Dictionary with model evaluation results
    """
    if input_directory is None:
        raise ValueError("input_directory must be provided")
        
    trained_models = [f for f in os.listdir(model_folder) if f.endswith('.keras')]
    results = {}
    
    for model_name in trained_models:
        print(f"\n--- Processing model: {model_name} ---")
        
        # Extract parameters from filename
        lookback, forecast = extract_model_params(model_name)
        
        if lookback is None or forecast is None:
            print(f"Skipping {model_name} - could not extract parameters")
            continue
            
        print(f"Extracted parameters - Lookback: {lookback}, Forecast: {forecast}")
        
        try:
            # Load model
            model_path = os.path.join(model_folder, model_name)
            model = tf.keras.models.load_model(model_path, compile=True)
            
            # Import data_preparation here to avoid circular imports
            from utils import data_preparation
            
            # Prepare data with extracted parameters
            X_test, Y_test = data_preparation.prepare_data(
                input_directory, code, lookback, forecast, 
                debug=True, univariate=True, scaler=scaler, train=False, 
                covid_token=False, cutoff_date=default_config.cutoff_date, 
                max_date=default_config.final_cutoff_date, 
                eliminate_covid_data=default_config.eliminate_covid_data, 
                covid_dates=default_config.covid_dates,
                split_ratio=default_config.default_split_ratio,
            )
            
            # Evaluate model
            loss, mae, mse = model.evaluate(X_test, Y_test, verbose=0)
            
            results[model_name] = {
                'lookback': lookback,
                'forecast': forecast,
                'loss': loss,
                'mae': mae,
                'mse': mse
            }
            
            print(f"Results - Loss: {loss:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}")
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
            continue
    
    return results


def create_pandemic_waves_df():
    """
    Create a DataFrame with pandemic wave periods.
    
    Returns:
        pd.DataFrame: DataFrame with wave names and date ranges
    """
    waves = {
        "Primera Onada": ("2020-03", "2020-06"),
        "Segona Onada": ("2020-10", "2020-12"),
        "Tercera Onada": ("2021-01", "2021-03"),
        "Quarta Onada": ("2021-04", "2021-06"),
        "Cinquena Onada": ("2021-07", "2021-09")
    }

    # Convert to DataFrame
    df_waves = pd.DataFrame(waves).T.reset_index()
    df_waves.columns = ["Onada", "Inici", "Final"]
    df_waves["Inici"] = pd.to_datetime(df_waves["Inici"])
    df_waves["Final"] = pd.to_datetime(df_waves["Final"])
    
    return df_waves



def setup_gpu_memory():
    """
    Configure GPU memory growth to prevent TensorFlow from allocating all GPU memory at once.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU memory setup error: {e}")
    else:
        print("No GPUs found, using CPU")


def create_model_directories():
    """
    Create necessary directories for model storage and plots.
    """
    directories = ['models', 'plots', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created/verified directory: {directory}")


def get_model_summary_info(model):
    """
    Extract key information from a Keras model.
    
    Args:
        model: Keras model
        
    Returns:
        dict: Dictionary with model information
    """
    return {
        'total_params': model.count_params(),
        'trainable_params': sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
        'non_trainable_params': sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights]),
        'layers': len(model.layers),
        'input_shape': model.input_shape,
        'output_shape': model.output_shape
    }


def save_results_to_csv(results, filename='model_evaluation_results.csv'):
    """
    Save model evaluation results to CSV file.
    
    Args:
        results: Dictionary with model evaluation results
        filename: Name of the output CSV file
    """
    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results.index.name = 'model_name'
    df_results.to_csv(filename)
    print(f"Results saved to {filename}")


def load_and_preprocess_data(data_path, target_code="T14", date_cutoff='2010-01-01'):
    """
    Load and preprocess the raw diagnostic data.
    
    Args:
        data_path: Path to the CSV data file
        target_code: Target diagnostic code to extract
        date_cutoff: Date to filter data from
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    # Load data
    df = smart_read(data_path, index_col=0)
    
    # Create COVID-19 combined code
    if 'B34' in df.columns and 'U07' in df.columns:
        df['COV-19'] = df["B34"] + df["U07"]
        df = df.drop(['B34', 'U07'], axis=1)
    
    # Convert index to datetime
    df.index = pd.to_datetime(df.index)
    
    # Filter by date
    date_filter = pd.to_datetime(date_cutoff)
    df = df[df.index >= date_filter]
    
    # Create overall sum column
    df["Overall"] = df.iloc[:, 1:].sum(axis=1)
    
    return df


def calculate_forecast_metrics(y_true, y_pred):
    """
    Calculate comprehensive forecast evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        dict: Dictionary with various metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # WAPE (Weighted Absolute Percentage Error)
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
    
    # R-squared
    r2 = r2_score(y_true, y_pred)
    
    # Directional accuracy (for time series)
    y_true_diff = np.diff(y_true.flatten())
    y_pred_diff = np.diff(y_pred.flatten())
    directional_accuracy = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff)) * 100
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'WAPE': wape,
        'R2': r2,
        'Directional_Accuracy': directional_accuracy
    }

def load_mlflow_model_history(model_name, model_type="univariate_transformer"):

    """
    Load training history from an MLflow Keras model.

    Args:
        model: MLflow Keras model
    Returns:
        dict: Training history  
    """
    raw_model_name = model_name.replace(".keras", "")
    history_path = f"../history/{raw_model_name}_history.pkl"

    if os.path.exists(history_path):
        print(f" Found saved history at: {history_path}")

        # Load the history
        with open(history_path, "rb") as f:
            history_data = pickle.load(f)
        
        # Convert to DataFrame for easier handling
        history_df = pd.DataFrame(history_data)
        history_df["epoch"] = range(1, len(history_df) + 1)

        
        # --- Log metrics ---
        for epoch, row in history_df.iterrows():
            for metric, value in row.items():
                if metric != "epoch":
                    from src.utils.mlflow_logger import MLflowLogger
                    MLflowLogger(active=True).log_metric(metric + "_" + model_type, float(value), step=int(row["epoch"]))
        
        # --- Create and log plots ---
        metric_groups = {
            "loss": ["loss", "val_loss"],
            "accuracy": ["accuracy", "val_accuracy"],
        }

        for group_name, keys in metric_groups.items():
            available = [k for k in keys if k in history_df.columns]
            if not available:
                continue

            plt.figure(figsize=(8, 4))
            for k in available:
                plt.plot(history_df["epoch"], history_df[k], label=k+ "_" + model_type, linewidth=2)
            plt.xlabel("Epoch")
            plt.ylabel(group_name.capitalize())
            plt.title(f"Training vs Validation {group_name.capitalize()}")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()

            plot_path = f"../{PLOTS_DIR}/{model_name}_{group_name}_curve.png"
            
            plt.close()
            os.makedirs('../' + PLOTS_DIR, exist_ok=True)
            plt.savefig(plot_path)
            # Log as artifact
            from src.utils.mlflow_logger import MLflowLogger
            MLflowLogger(active=True).log_artifact(plot_path, artifact_path="plots")

            print("✅ History loaded and logged to MLflow successfully.")
    else:
        print(f"⚠️ No history file found at {history_path}")
    
