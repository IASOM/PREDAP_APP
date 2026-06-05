# preprocessing_lmlr.py
# ------------------------------------------------------
# Data preprocessing utilities for LMLR models
# Author: Guillem Hernández Guillamet
# Version: 1.0
# Date: 04/06/2025
# ------------------------------------------------------

import pandas as pd
import numpy as np


def smoother(df, window_size):
    """
    Apply moving average smoothing with expanding window for initial values.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input time series data
    window_size : int
        Size of the moving window
        
    Returns:
    --------
    pd.DataFrame
        Smoothed time series data
    """
    smoothed = {}
    for column in df.columns:
        column_list = []
        for i in range(len(df)):
            if i < window_size:
                column_list.append(df[column].iloc[:i+1].mean())
            else:
                column_list.append(df[column].iloc[i-window_size:i+1].mean())
        smoothed[column] = column_list
    smoothed_df = pd.DataFrame.from_dict(smoothed)
    smoothed_df.set_index(df.index, inplace=True)
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


def infection_index_df(df, prior_days):
    """
    Calculate infection index based on ratio to previous period sum.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input time series data
    prior_days : int
        Number of previous days to use for comparison
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with infection indices
    """
    infection_index = {}
    for column in df.columns:
        column_list = [
            float('nan') if i < prior_days 
            else df[column].iloc[i] / max(df[column].iloc[(i-prior_days):(i-1)].sum(), 1)
            for i in range(len(df))
        ]
        infection_index[column] = column_list
    infection_index = pd.DataFrame.from_dict(infection_index)
    infection_index.set_index(df.index, inplace=True)
    return infection_index