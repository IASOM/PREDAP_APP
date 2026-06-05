# stationarity_gcausal.py
# ------------------------------------------------------
# Stationarity testing utilities for Granger causality analysis
# Author: Guillem Hernández Guillamet
# Version: 1.0
# Date: 04/06/2025
# ------------------------------------------------------

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import kpss


def kpss_test(data_df):
    """
    Perform KPSS (Kwiatkowski-Phillips-Schmidt-Shin) stationarity test on time series.
    
    The KPSS test is used to test the null hypothesis that a time series is stationary
    around a deterministic trend against the alternative that it has a unit root.
    
    Parameters:
    -----------
    data_df : pd.DataFrame
        Time series data to test
        
    Returns:
    --------
    pd.DataFrame
        Transposed DataFrame with test results:
        - Test statistic: KPSS test statistic
        - p-value: Statistical significance (< 0.05 suggests non-stationarity)
        - Critical values: Threshold values for different significance levels
        
    Notes:
    ------
    - Low p-value (< 0.05) indicates NON-STATIONARY series (reject null hypothesis)
    - High p-value (> 0.05) indicates STATIONARY series (fail to reject null)
    - Uses 'ct' regression (constant and trend) for detrending
    - Skips series with insufficient variation or length
    """
    test_stat, p_val, cv_1, cv_2_5, cv_5, cv_10 = [], [], [], [], [], []
    valid_cols = []
    
    for c in data_df.columns:
        series = data_df[c].dropna()
        
        # Skip series with insufficient data or no variation
        if series.nunique() <= 1 or len(series) < 10:
            print(f"Warning: Skipping column '{c}' - insufficient data or no variation")
            continue
            
        try:
            # Perform KPSS test with constant and trend
            res = kpss(series, regression='ct', nlags='auto')
            
            test_stat.append(res[0])
            p_val.append(res[1])
            cv_1.append(res[3]['1%'])
            cv_2_5.append(res[3]['2.5%'])
            cv_5.append(res[3]['5%'])
            cv_10.append(res[3]['10%'])
            valid_cols.append(c)
            
        except ValueError as e:
            print(f"Warning: KPSS test failed for column '{c}': {e}")
            continue
    
    if not valid_cols:
        print("Warning: No valid columns for KPSS testing")
        return pd.DataFrame()
    
    results_df = pd.DataFrame({
        'Test statistic': test_stat,
        'p-value': p_val,
        'Critical value - 1%': cv_1,
        'Critical value - 2.5%': cv_2_5,
        'Critical value - 5%': cv_5,
        'Critical value - 10%': cv_10
    }, index=valid_cols)
    
    return results_df.T.round(4)


def identify_nonstationary_series(kpss_results, significance_level=0.05):
    """
    Identify non-stationary series based on KPSS test results.
    
    Parameters:
    -----------
    kpss_results : pd.DataFrame
        Results from kpss_test() function
    significance_level : float, default=0.05
        Significance level for the test
        
    Returns:
    --------
    list
        List of column names that are non-stationary (p-value < significance_level)
    """
    if kpss_results.empty:
        return []
    
    # Get p-values row
    p_values = kpss_results.loc['p-value']
    
    # Find series with p-value < significance_level (non-stationary)
    nonstationary_series = p_values[p_values < significance_level].index.tolist()
    
    print(f"Non-stationary series (p-value < {significance_level}): {nonstationary_series}")
    
    return nonstationary_series


def stationarity_summary(kpss_results):
    """
    Provide a summary of stationarity test results.
    
    Parameters:
    -----------
    kpss_results : pd.DataFrame
        Results from kpss_test() function
        
    Returns:
    --------
    dict
        Summary statistics about stationarity
    """
    if kpss_results.empty:
        return {"total_series": 0, "stationary": 0, "non_stationary": 0}
    
    p_values = kpss_results.loc['p-value']
    total_series = len(p_values)
    non_stationary = (p_values < 0.05).sum()
    stationary = total_series - non_stationary
    
    summary = {
        "total_series": total_series,
        "stationary": stationary,
        "non_stationary": non_stationary,
        "percent_stationary": (stationary / total_series) * 100
    }
    
    print(f"Stationarity Summary:")
    print(f"  Total series: {total_series}")
    print(f"  Stationary: {stationary} ({summary['percent_stationary']:.1f}%)")
    print(f"  Non-stationary: {non_stationary} ({100 - summary['percent_stationary']:.1f}%)")
    
    return summary