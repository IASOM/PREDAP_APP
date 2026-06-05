# evaluation_dl.py
# =================
# Model evaluation utilities for deep learning models
# Author: Guillem Hernández Guillamet
# Version: 1.0

import tensorflow as tf


def evaluate_forecast(y_test_inverse, yhat_inverse):
    """
    Evaluate forecast performance using multiple metrics.
    
    Parameters:
    -----------
    y_test_inverse : np.ndarray
        True values (inverse transformed)
    yhat_inverse : np.ndarray
        Predicted values (inverse transformed)
        
    Example:
    --------
    >>> evaluate_forecast(y_true, y_pred)
    mae: 2.45
    mse: 8.12
    mape: 15.3
    """
    try:
        # Define metric functions
        mse_ = tf.keras.losses.MeanSquaredError()
        mae_ = tf.keras.losses.MeanAbsoluteError()
        mape_ = tf.keras.losses.MeanAbsolutePercentageError() 
        
        # Calculate metrics
        mae = mae_(y_test_inverse, yhat_inverse)
        print('mae:', mae)
        
        mse = mse_(y_test_inverse, yhat_inverse)
        print('mse:', mse)
        
        mape = mape_(y_test_inverse, yhat_inverse)
        print('mape:', mape)
        
    except Exception as e:
        print(f"Error during forecast evaluation: {str(e)}")


def get_results(y_test_inverse, yhat_inverse):
    """
    Calculate and return forecast performance metrics.
    
    Parameters:
    -----------
    y_test_inverse : np.ndarray
        True values (inverse transformed)
    yhat_inverse : np.ndarray
        Predicted values (inverse transformed)
        
    Returns:
    --------
    tuple
        MAE, MSE, and MAPE values
        
    Example:
    --------
    >>> mae, mse, mape = get_results(y_true, y_pred)
    >>> print(f"MAE: {mae:.3f}, MSE: {mse:.3f}, MAPE: {mape:.3f}")
    """
    try:
        # Define metric functions
        mse_ = tf.keras.losses.MeanSquaredError()
        mae_ = tf.keras.losses.MeanAbsoluteError()
        mape_ = tf.keras.losses.MeanAbsolutePercentageError() 
        
        # Calculate metrics
        mae = mae_(y_test_inverse, yhat_inverse)
        mse = mse_(y_test_inverse, yhat_inverse)
        mape = mape_(y_test_inverse, yhat_inverse)

        return mae, mse, mape
        
    except Exception as e:
        print(f"Error calculating results: {str(e)}")
        return None, None, None