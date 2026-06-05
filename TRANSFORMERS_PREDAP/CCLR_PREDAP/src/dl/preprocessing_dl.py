# preprocessing_dl.py
# ===================
# Data preprocessing utilities for deep learning models
# Author: Guillem Hernández Guillamet
# Version: 1.0

import pandas as pd
import numpy as np


def add_temprality(subdf, cc_predictors):
    """
    Add temporal features (weekday and month) to the DataFrame.
    
    Parameters:
    -----------
    subdf : pd.DataFrame
        Input DataFrame with datetime index
    cc_predictors : list
        List of predictor column names
        
    Returns:
    --------
    tuple
        Updated DataFrame and predictor list with temporal features
        
    Example:
    --------
    >>> df_with_time, predictors = add_temprality(df, ['var1', 'var2'])
    """
    try:
        # Extract temporal information from index
        subdf['weekday'] = subdf.index.day_name()    # Monday, Tuesday, etc.
        subdf['month'] = subdf.index.month_name()    # January, February, etc.

        # One-hot encode the categorical variables
        dummies_day = pd.get_dummies(subdf['weekday'], prefix='weekday_')
        dummies_month = pd.get_dummies(subdf['month'], prefix='month_')

        # Convert to integer type
        dummies_day = dummies_day.astype(int)
        dummies_month = dummies_month.astype(int)

        # Concatenate the dummy variables to the original DataFrame
        subdf = pd.concat([subdf, dummies_day, dummies_month], axis=1)

        # Drop the original categorical columns
        subdf = subdf.drop(['weekday', 'month'], axis=1)

        # Update cc_predictors list
        cc_predictors = cc_predictors + dummies_day.columns.tolist()
        cc_predictors = cc_predictors + dummies_month.columns.tolist()
        
        # Remove 'date' if it exists in cc_predictors
        if 'date' in cc_predictors:
            cc_predictors.remove('date')

        return subdf, cc_predictors
        
    except Exception as e:
        print(f"Error in add_temprality: {str(e)}")
        return subdf, cc_predictors


def split_sequence(sequence, look_back, forecast_horizon):
    """
    Split the time series into chunks of the correct size for supervised learning.
    
    Parameters:
    -----------
    sequence : np.ndarray
        Input time series data
    look_back : int
        Number of time steps to look back (input sequence length)
    forecast_horizon : int
        Number of time steps to forecast (output sequence length)
        
    Returns:
    --------
    tuple
        X (input sequences), y (target sequences)
        
    Example:
    --------
    >>> X, y = split_sequence(data, look_back=30, forecast_horizon=7)
    >>> print(f"Input shape: {X.shape}, Target shape: {y.shape}")
    """
    try:
        X, y = list(), list()
        
        for i in range(len(sequence)): 
            lag_end = i + look_back
            forecast_end = lag_end + forecast_horizon
            
            if forecast_end > len(sequence):
                break
                
            seq_x, seq_y = sequence[i:lag_end], sequence[lag_end:forecast_end]
            X.append(seq_x)
            y.append(seq_y)
            
        return np.array(X), np.array(y)
        
    except Exception as e:
        print(f"Error in split_sequence: {str(e)}")
        return np.array([]), np.array([])