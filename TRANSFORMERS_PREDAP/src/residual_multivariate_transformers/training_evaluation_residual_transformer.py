"""
Training and Evaluation Module for Residual Multivariate Transformers
=====================================================================

This module contains functions for training and evaluating residual multivariate
transformer models, including model training, GPU memory management, and callbacks.
"""

import os
import pickle
import sys
import tensorflow as tf
from datetime import datetime
import json
import pandas as pd
import numpy as np

from src.core.layers import PositionalEncoding, RevIN, CustomCosineDecay


# Add the src directory to path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir)) if 'residual_multivariate_transformers' in current_dir else os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


from src.core.config_manager import get_config

default_config = get_config()


def setup_gpu_memory():
    """
    Configure GPU memory growth to prevent TensorFlow from allocating all GPU memory.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU memory growth enabled for {gpu}")
        except RuntimeError as e:
            print(f"Error setting GPU memory growth: {e}")
    else:
        print("No GPUs found. Running on CPU.")


def train_given_model_and_data(model, X, Y, 
                               batch_size=None,
                               model_name=None, 
                               epochs=None,
                               validation_split=None,
                               shuffle=None,
                               patience=None,
                               save_history=None, 
                               save_model=None, 
                               save_memory=None, 
                               callbacks=None,
                               model_folder=None):
    """
    Train a model with given data and parameters.
    
    Parameters:
    -----------
    model : tf.keras.Model
        The model to train
    X : np.ndarray
        Training input data
    Y : np.ndarray
        Training target data
    batch_size : int, optional
        Batch size for training (default from config)
    model_name : str, optional
        Name for saving the model (default: "testing")
    epochs : int, optional
        Number of training epochs (default from config)
    validation_split : float, optional
        Fraction of data to use for validation (default from config)
    shuffle : bool, optional
        Whether to shuffle training data (default from config)
    patience : int, optional
        Early stopping patience (default from config)
    save_history : bool, optional
        Whether to save training history (default from config)
    save_model : bool, optional
        Whether to save the trained model (default from config)
    save_memory : bool, optional
        Whether to log GPU memory usage (default from config)
    callbacks : list, optional
        List of Keras callbacks to use during training
        
    Returns:
    --------
    tf.keras.callbacks.History or None
        Training history if model was trained, None if model already exists
    """
    # Use default parameters if not provided
    if batch_size is None:
        batch_size = default_config.DEFAULT_RESIDUAL_TRAINING_PARAMS['batch_size']
    if epochs is None:
        epochs = default_config.DEFAULT_RESIDUAL_TRAINING_PARAMS['epochs']
    if validation_split is None:
        validation_split = default_config.DEFAULT_RESIDUAL_TRAINING_PARAMS['validation_split']
    if shuffle is None:
        shuffle = default_config.DEFAULT_RESIDUAL_TRAINING_PARAMS['shuffle']
    if patience is None:
        patience = default_config.DEFAULT_RESIDUAL_TRAINING_PARAMS['patience']
    if save_history is None:
        save_history = default_config.DEFAULT_RESIDUAL_SAVE_PARAMS['save_history']
    if save_model is None:
        save_model = default_config.DEFAULT_RESIDUAL_SAVE_PARAMS['save_model']
    if save_memory is None:
        save_memory = default_config.DEFAULT_RESIDUAL_SAVE_PARAMS['save_memory']
    
    if model_name is None:
        model_name = "testing"

    # Configure GPU memory growth if save_memory is enabled
    if save_memory:
        setup_gpu_memory()

    # Define default callbacks if none provided
    if callbacks is None:
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            mode='min', 
            patience=patience, 
            restore_best_weights=True
        )
        callbacks = [early_stop]

    save_folder = model_folder or default_config.model_folder
    save_path = os.path.join(save_folder, model_name)
    if os.path.exists(save_path):
        print(f"Model {model_name} already exists at {save_path}")
        return None

    # Train the model
    print(f"Starting training for model: {model_name}")
    print(f"Training parameters:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Validation split: {validation_split}")
    print(f"  - Shuffle: {shuffle}")
    
    history = model.fit(
        x=X, 
        y=Y, 
        batch_size=batch_size,
        epochs=epochs, 
        shuffle=shuffle,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )

    # Save training history
    if save_history:
        raw_model_name = model_name.replace('.keras', '')
        history_filename = f'{raw_model_name}_history.pkl'
        with open('../history/' + history_filename, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        print(f"Training history saved to: {history_filename}")
    
    # Save model
    if save_model:
        save_folder = model_folder or default_config.model_folder
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, model_name)
        model.save(save_path)
        print(f"Model saved to: {save_path}")
        
    # Log memory usage if enabled
    if save_memory:
        try:
            memory_info = tf.config.experimental.get_memory_info('GPU:0')
            with open(default_config.MEMORY_LOG_FILE, 'a') as resultcsv:
                resultcsv.write(f"{model_name},{memory_info['peak']},train\n")
            print(f"Current memory usage: {memory_info['current'] / (batch_size**2)} MB")
            print(f"Peak memory usage: {memory_info['peak'] / (batch_size**2)} MB")
        except Exception as e:
            print(f"Could not log memory usage: {e}")

    return history


def evaluate_model(model, X_test, Y_test, verbose=1):
    """
    Evaluate a trained model on test data.
    
    Parameters:
    -----------
    model : tf.keras.Model
        The trained model to evaluate
    X_test : np.ndarray
        Test input data
    Y_test : np.ndarray
        Test target data
    verbose : int, optional
        Verbosity level for evaluation (default: 1)
        
    Returns:
    --------
    tuple
        (loss, mae, mse) evaluation metrics
    """
    print("Evaluating model on test data...")
    results = model.evaluate(X_test, Y_test, verbose=verbose)
    
    if len(results) == 3:
        loss, mae, mse = results
        print(f"Test Results - Loss: {loss:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}")
        return loss, mae, mse
    else:
        print(f"Test Results - Loss: {results[0]:.4f}")
        return results


def load_trained_model(model_path):
    """
    Load a trained model from disk.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
        
    Returns:
    --------
    tf.keras.Model
        Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    

    
    print(f"Loading model from: {model_path}")
    tf.keras.backend.clear_session()

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.reset_memory_stats(gpu)
        except Exception:
            pass

    model = tf.keras.models.load_model(
        model_path, 
        custom_objects={
            'CustomCosineDecay': CustomCosineDecay, 
            'PositionalEncoding': PositionalEncoding, 
            'RevIN': RevIN}, 
        compile=False,
            )
    
    print("Model loaded successfully")
    return model


def create_model_directories():
    """
    Create necessary directories for model storage.
    """
    directories = ['models', 'plots', 'logs']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")


def get_callbacks(patience=None, monitor='val_loss', mode='min', 
                  restore_best_weights=True, additional_callbacks=None):
    """
    Create a list of callbacks for model training.
    
    Parameters:
    -----------
    patience : int, optional
        Early stopping patience (default from config)
    monitor : str, optional
        Metric to monitor for early stopping (default: 'val_loss')
    mode : str, optional
        Mode for monitoring metric (default: 'min')
    restore_best_weights : bool, optional
        Whether to restore best weights (default: True)
    additional_callbacks : list, optional
        Additional callbacks to include
        
    Returns:
    --------
    list
        List of Keras callbacks
    """
    if patience is None:
        patience = default_config.DEFAULT_RESIDUAL_TRAINING_PARAMS['patience']
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=patience,
            restore_best_weights=restore_best_weights
        )
    ]
    
    if additional_callbacks:
        callbacks.extend(additional_callbacks)
    
    return callbacks


def save_performance_results(model_name, original_mae, original_mse, original_rmse, original_wape,
                           corrected_mae, corrected_mse, corrected_rmse, corrected_wape,
                           forecast, lookback, code, output_dir="results"):
    """
    Save performance comparison results to a JSON file.
    
    Parameters:
    -----------
    model_name : str
        Name of the residual model being evaluated
    original_mae, original_mse, original_rmse : float
        Performance metrics for the original base model
    corrected_mae, corrected_mse, corrected_rmse : float
        Performance metrics for the residual-corrected model
    forecast : int
        Forecast horizon used
    lookback : int
        Lookback window used
    code : str
        Target diagnostic code
    output_dir : str
        Directory to save the results file
    
    Returns:
    --------
    str
        Path to the saved JSON file
    """
    
    # Create output directory if it doesn't exist
    if not os.path.exists('../' +output_dir):
        os.makedirs('../' + output_dir)
        print(f"Created directory: {output_dir}")
    
    # Calculate improvements
    mae_improvement = float((original_mae - corrected_mae) / original_mae * 100) if original_mae != 0 else 0
    mse_improvement = float((original_mse - corrected_mse) / original_mse * 100) if original_mse != 0 else 0
    rmse_improvement =float((original_rmse - corrected_rmse) / original_rmse * 100) if original_rmse != 0 else 0
    wape_improvement =float((original_wape - corrected_wape) / original_wape * 100) if original_wape != 0 else 0
    
    # Create results dictionary
    results = {
        "model_info": {
            "residual_model_name": model_name,
            "target_code": code,
            "forecast_horizon": forecast,
            "lookback_window": lookback,
            "evaluation_timestamp": datetime.now().isoformat(),
            "model_type": "Residual Multivariate Transformer"
        },
        "original_model_performance": {
            "MAE": round(float(original_mae), 6),
            "MSE": round(float(original_mse), 6),
            "RMSE": round(float(original_rmse), 6),
            "WAPE": round(float(original_wape), 6)
        },
        "corrected_model_performance": {
            "MAE": round(float(corrected_mae), 6),
            "MSE": round(float(corrected_mse), 6),
            "RMSE": round(float(corrected_rmse), 6),
            "WAPE": round(float(corrected_wape), 6)

        },
        "improvements": {
            "MAE_improvement_percent": round(float(mae_improvement), 2),
            "MSE_improvement_percent": round(float(mse_improvement), 2),
            "RMSE_improvement_percent": round(float(rmse_improvement), 2),
            "WAPE_improvement_percent": round(float(wape_improvement), 2),
            "overall_assessment": "positive" if float(mae_improvement) > 0 else "negative"
        },
        "summary": {
            "best_metric": "MAE" if abs(mae_improvement) >= max(abs(mse_improvement), abs(rmse_improvement)) else 
                         "MSE" if abs(mse_improvement) >= abs(rmse_improvement) else "RMSE",
            "best_improvement": max(mae_improvement, mse_improvement, rmse_improvement),
            "average_improvement": round((mae_improvement + mse_improvement + rmse_improvement) / 3, 2)
        }
    }
    
    # Generate filename based on model name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_model_name = model_name.replace('.keras', '').replace('/', '_').replace('\\', '_')
    filename = f"performance_{clean_model_name}_{timestamp}.json"
    filepath = os.path.join('../' +output_dir, filename)
    
    # Save to JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\n📊 Performance results saved to: {filepath}")
    
    return filepath


def load_performance_results(results_dir="results"):
    """
    Load and display all performance results from JSON files.
    
    Parameters:
    -----------
    results_dir : str
        Directory containing the performance JSON files
        
    Returns:
    --------
    list
        List of loaded performance dictionaries
    """
    if not os.path.exists('../' +results_dir):
        print(f"Results directory '{results_dir}' not found.")
        return []
    
    json_files = [f for f in os.listdir('../' +results_dir) if f.endswith('.json') and f.startswith('performance_')]
    
    if not json_files:
        print(f"No performance JSON files found in '{results_dir}'.")
        return []
    
    results = []
    print(f"\n📊 Found {len(json_files)} performance result(s):")
    print("="*80)
    
    for json_file in sorted(json_files):
        filepath = os.path.join('../' +results_dir, json_file)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                result = json.load(f)
            results.append(result)
            
            # Display summary
            model_info = result['model_info']
            improvements = result['improvements']
            
            print(f"\n🔍 {model_info['residual_model_name']}")
            print(f"   Target: {model_info['target_code']} | "
                  f"Forecast: {model_info['forecast_horizon']} | "
                  f"Lookback: {model_info['lookback_window']}")
            print(f"   Evaluated: {model_info['evaluation_timestamp'][:19]}")
            print(f"   Improvements: MAE {improvements['MAE_improvement_percent']:+.2f}% | "
                  f"MSE {improvements['MSE_improvement_percent']:+.2f}% | "
                  f"RMSE {improvements['RMSE_improvement_percent']:+.2f}%")
            print(f"   Overall: {improvements['overall_assessment'].upper()} "
                  f"(Avg: {improvements['MAE_improvement_percent']:+.2f}%)")
            
        except Exception as e:
            print(f"❌ Error loading {json_file}: {e}")
    
    print("="*80)
    return results


def compare_model_performance(results_dir="../results", metric="MAE"):
    """
    Compare performance of multiple models and rank them.
    
    Parameters:
    -----------
    results_dir : str
        Directory containing performance JSON files
    metric : str
        Metric to use for comparison ('MAE', 'MSE', 'RMSE')
        
    Returns:
    --------
    pd.DataFrame
        Comparison table sorted by improvement
    """
    results = load_performance_results(results_dir)
    
    if not results:
        return None
    
    # Create comparison data
    comparison_data = []
    for result in results:
        model_info = result['model_info']
        improvements = result['improvements']
        
        comparison_data.append({
            'Model': model_info['residual_model_name'],
            'Target': model_info['target_code'],
            'Forecast': model_info['forecast_horizon'],
            'Lookback': model_info['lookback_window'],
            'MAE_Improvement_%': improvements['MAE_improvement_percent'],
            'MSE_Improvement_%': improvements['MSE_improvement_percent'],
            'RMSE_Improvement_%': improvements['RMSE_improvement_percent'],
            'Average_Improvement_%': (
                improvements['MAE_improvement_percent'] + 
                improvements['MSE_improvement_percent'] + 
                improvements['RMSE_improvement_percent']
            ) / 3,
            'Evaluation_Date': model_info['evaluation_timestamp'][:10]
        })
    
    # Create DataFrame and sort by selected metric
    df = pd.DataFrame(comparison_data)
    sort_column = f'{metric}_Improvement_%'
    df_sorted = df.sort_values(sort_column, ascending=False)
    
    print(f"\n🏆 MODEL PERFORMANCE RANKING (by {metric} improvement):")
    print("="*100)
    print(df_sorted.to_string(index=False))
    
    return df_sorted
