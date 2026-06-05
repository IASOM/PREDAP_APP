# preprocessing_gcausal.py
# ------------------------------------------------------
# Data preprocessing utilities for Granger causality analysis
# Author: Guillem Hernández Guillamet
# Version: 1.0
# Date: 04/06/2025
# ------------------------------------------------------

import pandas as pd
import numpy as np


def smoother(df, window_size):
    """
    Apply moving average smoothing to time series data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input time series data
    window_size : int
        Size of the moving window for averaging
        
    Returns:
    --------
    pd.DataFrame
        Smoothed time series data with same index as input
    """
    smoothed = {}
    for column in df.columns:
        column_list = [
            df[column].iloc[max(0, i - window_size):i + 1].mean() 
            for i in range(len(df))
        ]
        smoothed[column] = column_list
    smoothed_df = pd.DataFrame(smoothed, index=df.index)
    return smoothed_df


def min_max_scale(df):
    """
    Apply min-max normalization to DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data to normalize
        
    Returns:
    --------
    pd.DataFrame
        Normalized data with values between 0 and 1
    """
    return (df - df.min()) / (df.max() - df.min())

def stationate(df, cols):
    """
    Apply first-order differencing to make time series stationary.
    
    This function computes the first difference (y_t - y_{t-1}) for specified
    columns to remove trends and achieve stationarity.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input time series DataFrame
    cols : list
        List of column names to difference
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with differenced columns, NaN rows removed
        
    Notes:
    ------
    - First row will be dropped due to NaN from differencing
    - Original columns not in 'cols' remain unchanged
    - Use after KPSS test identifies non-stationary series
    """
    df_copy = df.copy()
    for col in cols:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col] - df_copy[col].shift(1)
    
    return df_copy.dropna()