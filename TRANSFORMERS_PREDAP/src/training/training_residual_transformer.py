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

from src.model_architechture.layers import PositionalEncoding, RevIN, CustomCosineDecay
from src.data_utils.column_mapping import resolve_columns


# Add the src directory to path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir)) if 'residual_multivariate_transformers' in current_dir else os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


from config.config_manager import get_config
from src.utils.environment_utils import setup_gpu_memory

default_config = get_config()


setup_gpu_memory()  # Configure GPU memory growth at the start of the module


def train_residual_model(model, X, Y, 
                               batch_size=None,
                               model_name=None, 
                               epochs=None,
                               validation_split=None,
                               shuffle=None,
                               patience=None,
                               save_history=None, 
                               save_model=None, 
                               save_memory=None, 
                               callbacks=None):
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

    # Check if model already exists
    if os.path.exists(f'{model_name}'):
        print(f"Model {model_name} already exists")
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
    if save_model and epochs > 1:
        os.makedirs(default_config.model_folder, exist_ok=True)
        model.save(os.path.join(default_config.model_folder, model_name))
        print(f"Model saved to: {model_name}")
        
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

def filter_diagnostics_covariates(df, diag_codes):
    """
    Filters the DataFrame to keep only the specified diagnostic codes.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input dataframe containing multiple codes
    diag_codes : list
        List of diagnostic codes to retain
        
    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame with only the specified diagnostic codes and timestamp
    """
    if 'timestamp' not in df.columns:
        raise KeyError("Expected a 'timestamp' column in the DataFrame.")
    
    # Ensure diag_codes is a list
    if isinstance(diag_codes, str):
        diag_codes = [diag_codes]
    
    resolved_codes, missing_codes = resolve_columns(
        df.columns,
        diag_codes,
        "diagnostic covariate columns",
    )
    cols_to_keep = ['timestamp'] + resolved_codes
    
    if len(cols_to_keep) <= 1:
        print("-> WARNING: No valid diagnostic covariates found in the DataFrame.")
        return df[['timestamp']].copy()
    
    filtered_df = df[cols_to_keep].copy()
    
    print(f"Filtered DataFrame to keep columns: {cols_to_keep}")
    
    return filtered_df



