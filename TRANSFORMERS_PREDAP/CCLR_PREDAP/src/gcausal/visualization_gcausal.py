# visualization_gcausal.py
# ------------------------------------------------------
# Visualization utilities for Granger causality analysis
# Author: Guillem Hernández Guillamet
# Version: 1.0
# Date: 04/06/2025
# ------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot


def lag_plots(data_df, show_plt = False):
    """
    Create lag plots for multiple time series to visualize autocorrelation patterns.
    
    A lag plot shows the relationship between y_t and y_{t+1}, helping identify:
    - Autocorrelation strength (diagonal patterns)
    - Non-linear relationships (curved patterns) 
    - Random behavior (scattered clouds)
    - Structural breaks or outliers
    
    Parameters:
    -----------
    data_df : pd.DataFrame
        Time series data to plot
        
    Returns:
    --------
    None
        Displays matplotlib figure with subplots
        
    Notes:
    ------
    - Creates one subplot per column in the DataFrame
    - Each subplot shows y_t on x-axis, y_{t+1} on y-axis
    - Useful for assessing stationarity before Granger causality tests
    """
    ncol = data_df.shape[1]
    
    # Handle empty DataFrame
    if ncol == 0:
        print("Warning: No columns to plot")
        return
    
    # Create subplot grid - single row with ncol columns
    fig, axes = plt.subplots(1, ncol, figsize=(5 * ncol, 5))
    
    # Handle single column case (axes is not a list)
    if ncol == 1:
        axes = [axes]
    
    # Create lag plot for each time series
    for i, col in enumerate(data_df.columns):
        lag_plot(data_df[col], ax=axes[i])
        axes[i].set_title(col, fontweight='bold', fontsize=12)
        axes[i].set_ylabel('$y_{t+1}$', fontweight='bold', fontsize=11)
        axes[i].set_xlabel('$y_t$', fontweight='bold', fontsize=11)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("plots/lag_plots.png")
    if show_plt:
        plt.show()
    plt.close()


def plot_var_metrics(metrics_df, show_plt = False):
    """
    Plot VAR model selection metrics (AIC, BIC, FPE, HQIC) vs lag order.
    
    Parameters:
    -----------
    metrics_df : pd.DataFrame
        DataFrame with columns ['AIC', 'BIC', 'FPE', 'HQIC'] and lag order as index
        
    Returns:
    --------
    None
        Displays matplotlib figure with 2x2 subplots
    """
    metrics_df.plot(subplots=True, marker='o', figsize=(15, 10), 
                    layout=(2, 2), sharex=True)
    plt.suptitle('VAR Model Selection Criteria', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("plots/var_metrics.png")
    if show_plt:
        plt.show()
    plt.close()


def plot_granger_matrix(causality_matrix, title="Granger Causality Matrix", cmap='RdYlBu_r', show_plt = False):
    """
    Visualize Granger causality matrix as a heatmap.
    
    Parameters:
    -----------
    causality_matrix : pd.DataFrame
        Square matrix with p-values from Granger causality tests
    title : str, default="Granger Causality Matrix"
        Title for the heatmap
    cmap : str, default='RdYlBu_r'
        Colormap for the heatmap
        
    Returns:
    --------
    None
        Displays matplotlib heatmap
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(causality_matrix.values, cmap=cmap, aspect='auto')
    plt.colorbar(label='P-value')
    plt.title(title, fontweight='bold', fontsize=14)
    plt.xlabel('Cause (X variables)', fontweight='bold')
    plt.ylabel('Effect (Y variables)', fontweight='bold')
    
    # Set tick labels
    plt.xticks(range(len(causality_matrix.columns)), causality_matrix.columns, rotation=45)
    plt.yticks(range(len(causality_matrix.index)), causality_matrix.index)
    
    # Add text annotations with p-values
    for i in range(len(causality_matrix.index)):
        for j in range(len(causality_matrix.columns)):
            plt.text(j, i, f'{causality_matrix.iloc[i, j]:.3f}', 
                    ha='center', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("plots/granger_causality_matrix.png")
    if show_plt:
        plt.show()
    plt.close()