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

from config.config_manager import get_config
from utils.experiments_utils import smart_read
from utils.environment_utils import setup_gpu_memory, create_model_directories

default_config = get_config()

def save_model_history(model_name, history):
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

def train_univariate_model(model, X, Y, batch_size=1024, model_name=None, epochs=100, 
                               save_history=False, save_model=True, save_memory=True, 
                               shuffle=False, callbacks=None, validation_split=0.3):
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
        validation_split: Proportion of data to use for validation
    """
    setup_gpu_memory()  # Configure GPU memory growth at the start of the function
    assert callbacks is not None, "Callbacks must be provided for training (e.g., EarlyStopping)"   

    if not os.path.exists(f'{model_name}'):  # check if model exist and run if not
        history = model.fit(x=X, 
                            y=Y, 
                            batch_size=batch_size,  # batch gradient descent (batch size 1024)
                            epochs=epochs, 
                            shuffle=shuffle,        # Allows shuffling
                            validation_split=validation_split,   # Proportion of data for validation
                            callbacks=callbacks)
    else:
        print(f"model {model_name} already exists")
        return None

    if save_history:  # save training history
        save_model_history(model_name, history)

    if save_model and epochs > 1:  # save model
        os.makedirs(default_config.model_folder, exist_ok=True)
        model.save(default_config.model_folder +"/" + model_name)

    return history


