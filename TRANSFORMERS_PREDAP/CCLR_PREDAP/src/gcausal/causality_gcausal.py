# causality_gcausal.py
# ------------------------------------------------------
# Granger causality testing utilities
# Author: Guillem Hernández Guillamet
# Version: 1.0
# Date: 04/06/2025
# ------------------------------------------------------

import pandas as pd
import numpy as np
import time
from multiprocessing import Pool, cpu_count
from functools import partial
from statsmodels.tsa.stattools import grangercausalitytests


def granger_causation_matrix(data, variables, p=1, test='ssr_chi2test'):
    """
    Compute a matrix of Granger causality test p-values between all variable pairs.
    
    Granger causality tests whether past values of one time series help predict
    another time series beyond what the second series' own past values can predict.


    Parameters:
    -----------
    data : pd.DataFrame
        Time series data (must be stationary)
    variables : list
        List of variable names to test
    p : int, default=1
        Maximum lag order to test
    test : str, default='ssr_chi2test'
        Statistical test to use. Options:
        - 'ssr_chi2test': Sum of squares residuals chi-square test
        - 'ssr_ftest': Sum of squares residuals F-test
        - 'lrtest': Likelihood ratio test  
        - 'params_ftest': Parameters F-test
        
    Returns:
    --------
    pd.DataFrame
        Square matrix where entry (i,j) is the p-value for testing whether
        variable j Granger-causes variable i. Lower p-values indicate stronger
        causal relationships.
        
    Notes:
    ------
    - Diagonal elements test self-causation (should be near 0)
    - P-value < 0.05 typically indicates significant Granger causality
    - Data should be stationary before testing
    - Requires sufficient observations relative to lag order
    """
    n_vars = len(variables)
    causality_matrix = pd.DataFrame(
        np.zeros((n_vars, n_vars)), 
        columns=variables, 
        index=variables
    )
    
    for cause_var in causality_matrix.columns:
        for effect_var in causality_matrix.index:
            try:
                # Select the two-variable system: [effect, cause]
                # Note: grangercausalitytests tests if cause_var Granger-causes effect_var
                test_data = data[[effect_var, cause_var]]
                
                # Run Granger causality test
                test_result = grangercausalitytests(test_data, p)#, verbose=False
                
                # Extract p-values for all lags tested
                p_values = [
                    round(test_result[lag+1][0][test][1], 4) 
                    for lag in range(p)
                ]
                
                # Use minimum p-value across all lags
                causality_matrix.loc[effect_var, cause_var] = np.min(p_values)
                
            except Exception as e:
                print(f"Warning: Granger test failed for {cause_var} -> {effect_var}: {e}")
                causality_matrix.loc[effect_var, cause_var] = np.nan
    
    # Rename columns and index for clarity
    causality_matrix.columns = [f"{var}_x" for var in variables]
    causality_matrix.index = [f"{var}_y" for var in variables]
    
    return causality_matrix


def interpret_causality_matrix(causality_matrix, significance_level=0.05, print_summary=True):
    """
    Interpret Granger causality results and identify significant relationships.
    
    Parameters:
    -----------
    causality_matrix : pd.DataFrame
        Output from granger_causation_matrix()
    significance_level : float, default=0.05
        P-value threshold for significance
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'significant_pairs': List of (cause, effect, p_value) tuples
        - 'summary': Summary statistics
        - 'strongest_effects': Top causal relationships
    """
    significant_pairs = []
    
    # Find significant causal relationships (excluding diagonal)
    for i, effect_var in enumerate(causality_matrix.index):
        for j, cause_var in enumerate(causality_matrix.columns):
            if i != j:  # Skip diagonal (self-causation)
                p_value = causality_matrix.iloc[i, j]
                if not np.isnan(p_value) and p_value < significance_level:
                    # Remove _x and _y suffixes for readability
                    clean_cause = cause_var.replace('_x', '')
                    clean_effect = effect_var.replace('_y', '')
                    significant_pairs.append((clean_cause, clean_effect, p_value))
    
    # Sort by p-value (strongest relationships first)
    significant_pairs.sort(key=lambda x: x[2])
    
    # Summary statistics
    total_tests = len(causality_matrix) * (len(causality_matrix.columns) - 1)  # Exclude diagonal
    n_significant = len(significant_pairs)
    
    summary = {
        'total_tests': total_tests,
        'significant_relationships': n_significant,
        'percent_significant': (n_significant / total_tests) * 100 if total_tests > 0 else 0
    }
    
    # Top relationships
    strongest_effects = significant_pairs[:5]  # Top 5
    
    results = {
        'significant_pairs': significant_pairs,
        'summary': summary,
        'strongest_effects': strongest_effects
    }
    
    if print_summary:
        # Print summary
        print(f"Granger Causality Analysis Summary:")
        print(f"  Total tests: {total_tests}")
        print(f"  Significant relationships (p < {significance_level}): {n_significant}")
        print(f"  Percentage significant: {summary['percent_significant']:.1f}%")
        
        if strongest_effects:
            print(f"\nTop {len(strongest_effects)} strongest causal relationships:")
            for cause, effect, p_val in strongest_effects:
                print(f"  {cause} → {effect} (p = {p_val:.4f})")
    
    return results


def filter_causality_matrix(causality_matrix, significance_level=0.05):
    """
    Filter causality matrix to show only significant relationships.
    
    Parameters:
    -----------
    causality_matrix : pd.DataFrame
        Output from granger_causation_matrix()
    significance_level : float, default=0.05
        P-value threshold for significance
        
    Returns:
    --------
    pd.DataFrame
        Matrix with only significant p-values, others set to NaN
    """
    filtered_matrix = causality_matrix.copy()
    filtered_matrix[filtered_matrix >= significance_level] = np.nan
    
    return filtered_matrix



def _granger_test_worker(args):
    """
    Worker function for parallel Granger causality testing.
    
    Parameters:
    -----------
    args : tuple
        (data, effect_var, cause_var, p, test)
        
    Returns:
    --------
    tuple
        (effect_var, cause_var, p_value_or_nan)
    """
    data, effect_var, cause_var, p, test = args
    
    try:
        # Select the two-variable system: [effect, cause]
        test_data = data[[effect_var, cause_var]]
        
        # Run Granger causality test
        test_result = grangercausalitytests(test_data, p)#, verbose=False
        
        # Extract p-values for all lags tested
        p_values = [
            round(test_result[lag+1][0][test][1], 4) 
            for lag in range(p)
        ]
        
        # Use minimum p-value across all lags
        min_p_value = np.min(p_values)
        
        return (effect_var, cause_var, min_p_value)
        
    except Exception as e:
        return (effect_var, cause_var, np.nan)


def granger_causation_matrix_parallel(data, variables, p=1, test='ssr_chi2test', n_processes=None):
    """
    Compute a matrix of Granger causality test p-values between all variable pairs using multiprocessing.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Time series data (must be stationary)
    variables : list
        List of variable names to test
    p : int, default=1
        Maximum lag order to test
    test : str, default='ssr_chi2test'
        Statistical test to use
    n_processes : int, optional
        Number of processes to use. If None, uses all available CPU cores.
        
    Returns:
    --------
    tuple
        (causality_matrix, execution_time)
        - causality_matrix: pd.DataFrame with p-values
        - execution_time: float, time in seconds
    """
    start_time = time.time()
    
    if n_processes is None:
        n_processes = cpu_count()
    
    print(f"Starting Granger causality analysis with {n_processes} processes...")
    print(f"Testing {len(variables)} variables ({len(variables)**2} total tests)")
    
    n_vars = len(variables)
    causality_matrix = pd.DataFrame(
        np.zeros((n_vars, n_vars)), 
        columns=variables, 
        index=variables
    )
    
    # Prepare arguments for all variable pairs
    test_args = []
    for cause_var in variables:
        for effect_var in variables:
            test_args.append((data, effect_var, cause_var, p, test))
    
    # Run tests in parallel
    with Pool(processes=n_processes) as pool:
        results = pool.map(_granger_test_worker, test_args)
    
    # Fill the causality matrix with results
    for effect_var, cause_var, p_value in results:
        causality_matrix.loc[effect_var, cause_var] = p_value
    
    # Rename columns and index for clarity
    causality_matrix.columns = [f"{var}_x" for var in variables]
    causality_matrix.index = [f"{var}_y" for var in variables]
    
    execution_time = time.time() - start_time
    
    print(f"Granger causality analysis completed in {execution_time:.2f} seconds")
    print(f"Average time per test: {execution_time/len(test_args):.4f} seconds")
    
    return causality_matrix
