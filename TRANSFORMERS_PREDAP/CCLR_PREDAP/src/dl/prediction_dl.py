# prediction_dl.py
# =================
# Prediction and transformation utilities for deep learning models
# Author: Guillem Hernández Guillamet
# Version: 1.0

import numpy as np


def prediction(model, X_test):
    """
    Generate predictions using a trained model.
    
    Parameters:
    -----------
    model : tensorflow.keras.Model
        Trained model for prediction
    X_test : np.ndarray
        Test input data
        
    Returns:
    --------
    np.ndarray
        Model predictions
        
    Example:
    --------
    >>> predictions = prediction(trained_model, X_test)
    """
    try:
        return model.predict(X_test)
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return np.array([])


def inverse_transform(y_true, y_pred, scaler):
    """
    Apply inverse transformation to scaled predictions and true values.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values (scaled)
    y_pred : np.ndarray
        Predicted values (scaled)
    scaler : sklearn.preprocessing.MinMaxScaler
        Fitted scaler object
        
    Returns:
    --------
    tuple
        Inverse transformed true values and predictions
        
    Example:
    --------
    >>> y_true_orig, y_pred_orig = inverse_transform(y_true, y_pred, scaler)
    """
    try:
        y_true_inv = scaler.inverse_transform(np.reshape(y_true, (-1, y_true.shape[-1])))
        y_pred_inv = scaler.inverse_transform(np.reshape(y_pred, (-1, y_pred.shape[-1])))
        return y_true_inv, y_pred_inv
    except Exception as e:
        print(f"Error during inverse transformation: {str(e)}")
        return y_true, y_pred