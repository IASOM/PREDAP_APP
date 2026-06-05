# model_selection_gcausal.py
# ------------------------------------------------------
# Model selection utilities for VAR and Granger causality analysis
# Author: Guillem Hernández Guillamet
# Version: 1.0
# Date: 04/06/2025
# ------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.var_model import VAR


def splitter(df, train_ratio=0.8):
    """
    Split time series data into training and testing sets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Time series data to split
    train_ratio : float, default=0.8
        Proportion of data to use for training
        
    Returns:
    --------
    tuple
        (train_df, test_df) - Training and testing DataFrames
    """
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"Data split: {len(train_df)} training samples, {len(test_df)} testing samples")
    
    return train_df, test_df


def select_p(train_df, lags_list, plot=True):
    """
    Select optimal lag order for VAR model using information criteria.
    
    This function fits VAR models with different lag orders and computes
    various information criteria to help select the optimal number of lags.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data for VAR model
    max_lags : int, default=59
        Maximum number of lags to test
    plot : bool, default=True
        Whether to plot the information criteria
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with information criteria for each lag order
        
    Notes:
    ------
    Information Criteria (lower is better):
    - AIC (Akaike): Balances fit and complexity, tends to select higher lags
    - BIC (Bayesian): More conservative, penalizes complexity more than AIC
    - FPE (Final Prediction Error): Focuses on prediction accuracy
    - HQIC (Hannan-Quinn): Between AIC and BIC in terms of penalty
    
    Recommendations:
    - For prediction: Use AIC or FPE
    - For model selection: Use BIC for more parsimonious models
    - For causality testing: Consider computational cost vs accuracy trade-off
    """
    # Initialize VAR model
    model = VAR(train_df)
    
    # Test range of lag orders
    if lags_list[-1] < len(train_df) // 2:
        p_range = lags_list
    else:
        raise ValueError("Lag list contains values too large for the dataset size.")
        
    metrics = {'AIC': [], 'BIC': [], 'FPE': [], 'HQIC': []}
    
    print(f"Testing lag orders from 1 to {max(p_range)}...")
    
    for p in p_range:
        try:
            # Fit VAR model with p lags
            var_result = model.fit(p)
            
            # Store information criteria
            metrics['AIC'].append(var_result.aic)
            metrics['BIC'].append(var_result.bic) 
            metrics['FPE'].append(var_result.fpe)
            metrics['HQIC'].append(var_result.hqic)
            
        except Exception as e:
            print(f"Warning: Failed to fit VAR model with {p} lags: {e}")
            # Fill with NaN for failed fits
            for metric in metrics:
                metrics[metric].append(np.nan)
    
    # Create results DataFrame
    results_df = pd.DataFrame(metrics, index=p_range)
    
    # Find optimal lags for each criterion
    optimal_lags = {}
    for criterion in metrics.keys():
        if not results_df[criterion].isna().all():
            optimal_lags[criterion] = results_df[criterion].idxmin()
        else:
            optimal_lags[criterion] = None
    
    print("Optimal lag orders:")
    for criterion, lag in optimal_lags.items():
        if lag is not None:
            print(f"  {criterion}: {lag} lags")
        else:
            print(f"  {criterion}: Unable to determine")
    
    # Plot results if requested
    if plot:
        _plot_information_criteria(results_df, optimal_lags)
    
    return results_df, optimal_lags


def _plot_information_criteria(results_df, optimal_lags):
    """
    Helper function to plot information criteria.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with information criteria
    optimal_lags : dict
        Dictionary with optimal lag orders for each criterion
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('VAR Model Information Criteria', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (criterion, color) in enumerate(zip(results_df.columns, colors)):
        ax = axes[i]
        
        # Plot criterion values
        ax.plot(results_df.index, results_df[criterion], 
               marker='o', color=color, linewidth=2, markersize=4)
        
        # Mark optimal lag
        if optimal_lags[criterion] is not None:
            optimal_value = results_df.loc[optimal_lags[criterion], criterion]
            ax.axvline(optimal_lags[criterion], color='red', linestyle='--', alpha=0.7)
            ax.plot(optimal_lags[criterion], optimal_value, 
                   marker='*', color='red', markersize=12)
            ax.text(optimal_lags[criterion], optimal_value, 
                   f'  Optimal: {optimal_lags[criterion]}', 
                   verticalalignment='center', fontweight='bold')
        
        ax.set_title(f'{criterion} (lower is better)', fontweight='bold')
        ax.set_xlabel('Lag Order')
        ax.set_ylabel(criterion)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    plt.close()


def fit_var_model(train_df, lag_order, verbose=True):
    """
    Fit a VAR model with specified lag order.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data
    lag_order : int
        Number of lags to include
    verbose : bool, default=True
        Whether to print model summary
        
    Returns:
    --------
    VARResults
        Fitted VAR model
    """
    #if some column in train_df is constant, put random values to it
    #Temporary fix for the error "ValueError: The VAR model is not identified. The number of lags is too large for the number of observations."
    
    for col in train_df.columns:
        if train_df[col].nunique() == 1:
            print(f"Warning: Column '{col}' is constant. Adding random noise to avoid VAR fitting issues.")
            train_df[col] = np.random.rand(len(train_df))
    model = VAR(train_df)
    
    var_result = model.fit(lag_order)
    
    
    if verbose:
        print(f"VAR Model Summary (lag order = {lag_order}):")
        print(f"  Number of observations: {var_result.nobs}")
        print(f"  Number of variables: {var_result.neqs}")
        print(f"  AIC: {var_result.aic:.4f}")
        print(f"  BIC: {var_result.bic:.4f}")
        print(f"  Log likelihood: {var_result.llf:.4f}")
    
    return var_result


def recommend_lag_order(optimal_lags, priority=['BIC', 'AIC', 'HQIC', 'FPE']):
    """
    Recommend a lag order based on multiple criteria with priority ordering.
    
    Parameters:
    -----------
    optimal_lags : dict
        Dictionary with optimal lag orders from select_p()
    priority : list, default=['BIC', 'AIC', 'HQIC', 'FPE']
        Priority order for criteria (first available is recommended)
        
    Returns:
    --------
    int
        Recommended lag order
    """
    for criterion in priority:
        if criterion in optimal_lags and optimal_lags[criterion] is not None:
            print(f"Recommended lag order: {optimal_lags[criterion]} (based on {criterion})")
            return optimal_lags[criterion]
    
    print("Warning: Unable to recommend lag order - all criteria failed")
    return 1  # Default fallback

def select_causal_features(causality_matrix, target_variable, significance_level=0.05):
    """
    Select features that Granger-cause the target variable based on significance level.
    
    Parameters:
    -----------
    causality_matrix : pd.DataFrame
        Output from granger_causation_matrix()
    target_variable : str
        The target variable to test for Granger causality
    significance_level : float, default=0.05
        P-value threshold for significance

    Returns:
    --------
    list
        List of variables that significantly Granger-cause the target variable
    """
    target_var = target_variable.replace('_x', '')  # Remove suffix for target variable
    significant_vars = []

    for cause_var in causality_matrix.columns:
        if cause_var != f"{target_var}_x":
            p_value = causality_matrix.loc[f"{target_var}_y", cause_var]
            if p_value < significance_level:
                significant_vars.append(cause_var.replace('_x', ''))

    print(f"Variables that Granger-cause {target_var} at p < {significance_level}: {significant_vars}")

    return significant_vars
