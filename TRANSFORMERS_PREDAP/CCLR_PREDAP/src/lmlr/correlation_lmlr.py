# correlation_lmlr.py
# ------------------------------------------------------
# Correlation analysis and multicollinearity utilities
# Author: Guillem Hernández Guillamet
# Version: 1.0
# Date: 04/06/2025
# ------------------------------------------------------

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor


def get_top_correlations_blog(df, threshold=0.90):
    """
    Find pairs of variables with correlation above threshold.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    threshold : float, default=0.90
        Correlation threshold for filtering
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with highly correlated variable pairs
    """
    orig_corr = df.corr()
    abs_corr = orig_corr.abs()
    so = abs_corr.unstack()
    pairs = set()
    result = pd.DataFrame(columns=['Variable 1', 'Variable 2', 'Correlation Coefficient'])
    
    for index, value in so.sort_values(ascending=False).items():
        if (value > threshold and 
            index[0] != index[1] and 
            (index[1], index[0]) not in pairs):
            result.loc[len(result)] = [index[0], index[1], orig_corr.loc[index[0], index[1]]]
            pairs.add((index[0], index[1]))

    result.columns = ['Variable 1', 'Variable 2', 'Correlation Coefficient']
    
    return result.set_index(['Variable 1', 'Variable 2'])


def compute_vif(variables, df):
    """
    Calculate Variance Inflation Factor for given variables.
    
    Parameters:
    -----------
    variables : list
        List of variable names
    df : pd.DataFrame
        Input DataFrame
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with VIF values for each variable
    """
    df = df.fillna(0)
    X = df[variables].copy()
    X['intercept'] = 1
    vif = pd.DataFrame()
    vif['Variable'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif[vif['Variable'] != 'intercept']

def compute_vif_matrix(variables, df, eps=1e-12):
    X = df[variables].select_dtypes(include=[np.number])#.dropna()

    # treu variança zero
    X = X.loc[:, X.var() > 1e-10]
    variables = X.columns.tolist()
    n_vars = len(variables)

    if n_vars <= 1:
        return pd.DataFrame({'Variable': variables, 'VIF': [1.0] * n_vars})

    # centra (això ja gestiona l'efecte de l'intercept)
    Xc = X.to_numpy(dtype=np.float64)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)

    # std per convertir a correlació (evita escales diferents)
    std = Xc.std(axis=0, ddof=0)
    keep = std > eps
    Xc = Xc[:, keep]
    variables = [v for v, k in zip(variables, keep) if k]
    n_vars = len(variables)

    Z = Xc / std[keep]

    # Matriu de correlació
    R = (Z.T @ Z) / (Z.shape[0] - 1)

    # Inversa (o pseudo-inversa si és singular)
    try:
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        R_inv = np.linalg.pinv(R)

    vif = np.diag(R_inv)
    # si surt negatiu per temes numèrics, ho clavem a inf/0 segons cas
    vif = np.where(vif > 0, vif, np.inf)

    return pd.DataFrame({'Variable': variables, 'VIF': vif})



def compute_vif_matrix_gpu(variables, df):
    """
    GPU-accelerated VIF calculation using CuPy.
    VIF_i = 1 / (1 - R²_i) where R²_i is from regressing X_i on other variables.
    
    Requires CuPy to be installed: pip install cupy-cuda12x (adjust for your CUDA version)
    
    Parameters:
    -----------
    variables : list
        List of variable names
    df : pd.DataFrame
        Input DataFrame
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with VIF values for each variable
        
    Raises:
    -------
    ImportError
        If CuPy is not installed
    RuntimeError
        If no compatible GPU is available
    """
    try:
        import cupy as cp
    except ImportError:
        raise ImportError(
            "CuPy is required for GPU acceleration. "
            "Install it with: pip install cupy-cuda12x (adjust for your CUDA version)"
        )
    
    X = df[variables].select_dtypes(include=[np.number]).dropna()
    
    # Remove variables with zero variance
    X = X.loc[:, X.var() > 1e-10]
    variables = X.columns.tolist()
    n_vars = len(variables)
    
    if n_vars <= 1:
        return pd.DataFrame({'Variable': variables, 'VIF': [1.0] * n_vars})

    # Transfer data to GPU
    X_matrix = cp.array(X.values, dtype=cp.float64)

    vif_values = []
    
    for i in range(n_vars):
        # Get dependent variable (column i)
        y = X_matrix[:, i]
        
        # Get independent variables (all columns except i)
        # Create index mask for all columns except i
        mask = cp.ones(n_vars, dtype=bool)
        mask[i] = False
        X_others = X_matrix[:, mask]
        
        # Add intercept to independent variables
        ones = cp.ones((X_others.shape[0], 1), dtype=cp.float64)
        X_others_with_intercept = cp.hstack([ones, X_others])
        
        try:
            # Solve least squares: β = (X'X)^(-1)X'y
            XtX = X_others_with_intercept.T @ X_others_with_intercept
            Xty = X_others_with_intercept.T @ y
            
            # Use pseudo-inverse for numerical stability
            beta = cp.linalg.pinv(XtX) @ Xty
            
            # Calculate predictions and R²
            y_pred = X_others_with_intercept @ beta
            ss_res = cp.sum((y - y_pred) ** 2)
            ss_tot = cp.sum((y - cp.mean(y)) ** 2)
            
            if ss_tot > 1e-10:
                r_squared = 1 - (ss_res / ss_tot)
                r_squared = cp.clip(r_squared, 0, 0.9999)  # Prevent division by zero
                vif = 1 / (1 - r_squared)
            else:
                vif = 1.0
                
        except (cp.linalg.LinAlgError, ValueError):
            vif = cp.inf
        
        # Transfer result back to CPU
        vif_values.append(float(cp.asnumpy(vif)))
    
    vif_df = pd.DataFrame({
        'Variable': variables,
        'VIF': vif_values
    })
    
    return vif_df


def approximate_vif_correlation(variables, df):
    """
    Fast VIF approximation using inverse correlation matrix diagonal elements.
    Note: This is an approximation, not the exact VIF formula.
    
    Parameters:
    -----------
    variables : list
        List of variable names
    df : pd.DataFrame
        Input DataFrame
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with approximate VIF values
    """
    X = df[variables].select_dtypes(include=[np.number]).dropna()
    X = X.loc[:, X.var() > 1e-10]
    variables = X.columns.tolist()
    
    # Calculate correlation matrix
    corr_matrix = X.corr().values
    
    try:
        # Diagonal elements of inverse correlation matrix
        inv_corr = np.linalg.pinv(corr_matrix)
        vif_values = np.diag(inv_corr)
        vif_values = np.maximum(vif_values, 1.0)  # VIF should be >= 1
        
    except np.linalg.LinAlgError:
        vif_values = np.ones(len(variables))
    
    return pd.DataFrame({
        'Variable': variables,
        'VIF': vif_values
    })

def filter_VIF(vif, df, iterations_max, VIF_threshold):
    """
    Filter variables by removing those with highest VIF iteratively.
    
    Parameters:
    -----------
    vif : pd.DataFrame
        VIF DataFrame
    df : pd.DataFrame
        Input DataFrame
    iterations_max : int
        Maximum number of iterations
    VIF_threshold : float
        VIF threshold for filtering
        
    Returns:
    --------
    list
        List of variables with VIF below threshold
    """
    # Start from the variables provided in the input VIF table
    redundant_vars = vif.sort_values('VIF', ascending=False)['Variable'].tolist()
    iterations = 0

    # Iteratively recompute VIF and drop the current worst offender
    while iterations < iterations_max and len(redundant_vars) > 1:
        vif_curr = compute_vif_matrix(redundant_vars, df)

        # In case compute_vif_matrix dropped constant columns etc.
        vars_curr = vif_curr['Variable'].tolist()
        if len(vars_curr) <= 1:
            return vars_curr

        max_vif = vif_curr['VIF'].max()
        if not np.isfinite(max_vif) or max_vif <= VIF_threshold:
            return vars_curr

        # Remove the variable with the highest CURRENT VIF
        worst_var = vif_curr.sort_values('VIF', ascending=False).iloc[0]['Variable']
        redundant_vars.remove(worst_var)

        iterations += 1

    return redundant_vars