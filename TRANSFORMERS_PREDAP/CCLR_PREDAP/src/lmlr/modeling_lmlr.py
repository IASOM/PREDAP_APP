# modeling_lmlr.py
# ------------------------------------------------------
# Model training and selection utilities
# Author: Guillem Hernández Guillamet
# Version: 1.0
# Date: 04/06/2025
# ------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from IPython.display import display

from .visualization_lmlr import plot_models, plot_metrics
from .metrics_lmlr import metrics_calculation


def models_training(df, code, corr, max_iters, plt_models = True, plt_metrics = True, plt_selected_models=True, plt_best_model=True):
    """
    Train multiple regression models with incremental feature addition.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data
    code : str
        Target variable name
    corr : pd.Series
        Correlation series for feature ordering
    max_iters : int
        Maximum number of models to train
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with model performance metrics and best model selection
    """
    print(f">>> Training models for predicting variable ... {code}")

    Y_PRED_train, Y_PRED_test, models, predictors = [], [], [], []
    corr = corr[:max_iters]
    split_idx = int(len(df) * 0.8)
    y_train = df[code].iloc[:split_idx]
    y_test = df[code].iloc[split_idx:]
    
    # Train models with incremental features
    for i in range(1, len(corr)):
        X_train = df[corr.index[:i]].iloc[:split_idx]
        X_test = df[corr.index[:i]].iloc[split_idx:]
        model = sm.OLS(y_train, X_train).fit()
        models.append(model)
        predictors.append(','.join(X_train.columns))
        Y_PRED_train.append(model.predict(X_train))
        Y_PRED_test.append(model.predict(X_test))

    # Plot all models
    if plt_models:
        plot_models(X_test, code, corr, Y_PRED_test, y_test)

    # Calculate metrics
    df_metrics = metrics_calculation(models, y_train, y_test, Y_PRED_train, Y_PRED_test, predictors)
    
    if plt_metrics:
        plot_metrics(df_metrics, code)

    # Select best models
    df_best, interesting_models = select_best_models(
        df_metrics, X_test, code, Y_PRED_test, y_test, pval=0.1, max_models=3, plt_selected_models=plt_selected_models
    )
    
    # Select absolute best model
    df_final = select_best_absolute_model(
        models, interesting_models, df_best, X_test, code, Y_PRED_test, y_test, plt_best_model=plt_best_model
    )
   
    return df_final


def select_best_models(df, df_test, code, Y_PRED_test, y_test, pval=0.1, max_models=3, plt_selected_models=True):
    """
    Select best models based on p-value threshold.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Model metrics DataFrame
    df_test : pd.DataFrame
        Test data
    code : str
        Target variable name
    Y_PRED_test : list
        Test predictions
    y_test : pd.Series
        Actual test values
    pval : float, default=0.1
        P-value threshold
    max_models : int, default=3
        Maximum number of models to select
        
    Returns:
    --------
    tuple
        (filtered_df, selected_model_indices)
    """
    interesting_models = []
    
    # Find models with significant p-values
    c = 0
    for i in range(len(df)):
        if df['pval1'].iloc[i] >= pval:  # Significant models
            c += 1
            if c <= max_models:
                interesting_models.append(i)
            else:
                pass
    
    if not interesting_models:
        print(f"Warning: No models found with p-value <= {pval}")
        # Fallback to best model by MAPE
        best_idx = df['MAPE_test'].idxmin()
        interesting_models = [best_idx]
        print(f"Selected best model by MAPE_test: index {best_idx}")
    
    # Filter DataFrame
    df_filtered = df.iloc[:max(interesting_models)+1].copy()
    
    print(">>> BEST INTERESTING MODELS -----------------------------------------")
    display(df_filtered)
    
    # Plot selected models
    if plt_selected_models:
        _plot_selected_models(df_test, code, Y_PRED_test, y_test, interesting_models)
    
    return df_filtered, interesting_models


def select_best_absolute_model(models, interesting_models, df, df_test, code, Y_PRED_test, y_test, mape=20.0, max_models=3, plt_best_model=True, plt_show=False):
    """
    Select and mark the absolute best model.
    
    Parameters:
    -----------
    models : list
        List of trained models
    interesting_models : list
        Indices of interesting models
    df : pd.DataFrame
        Model metrics DataFrame
    df_test : pd.DataFrame
        Test data
    code : str
        Target variable name
    Y_PRED_test : list
        Test predictions
    y_test : pd.Series
        Actual test values
    mape : float, default=20.0
        MAPE threshold (unused in current implementation)
    max_models : int, default=3
        Maximum models (unused in current implementation)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with best model marked
    """
    if not interesting_models:
        print("Warning: No interesting models to select the best from")
        return df
        
    best_model_idx = min(interesting_models)
    
    print(">>> ABSOLUTE BEST MODEL -----------------------------------------")
    print(models[best_model_idx].summary())    
    
    # Mark best model
    df['BEST_MODEL'] = "NO"
    df.loc[best_model_idx, "BEST_MODEL"] = "YES"

    if plt_best_model:

    
        # Plot best model
        fig, ax = plt.subplots(figsize=(15, 8))
        fig.suptitle(f'{code} prediction - Best Model', fontweight='bold', fontsize=14)
        
        
        ax.plot(df_test.index, y_test, 'go-', 
            label='Actual diagnoses', linewidth=2, markersize=4)

        ax.plot(df_test.index, Y_PRED_test[best_model_idx], 
            label=f'Best Model ({best_model_idx+1} vars)', linewidth=2)
        
        ax.set_xlabel('Date', fontweight='bold', fontsize=12)
        ax.set_ylabel('n diagnoses', fontweight='bold', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("plots/best_model.png")
        if plt_show:
            plt.show()
        
    print(df.iloc[:best_model_idx+1])
    
    return df


def _plot_selected_models(df_test, code, Y_PRED_test, y_test, selected_indices, plt_show=False):
    """
    Helper function to plot selected models.
    
    Parameters:
    -----------
    df_test : pd.DataFrame
        Test data with datetime index
    code : str
        Target variable name
    Y_PRED_test : list
        List of predictions
    y_test : pd.Series
        Actual test values
    selected_indices : list
        Indices of selected models
    """
    if not selected_indices:
        print("No models to plot")
        return
    
    plt.figure(figsize=(15, 8))
    plt.title(f'{code} Prediction - Selected Models', fontweight='bold', fontsize=14)
    
    # Plot actual values
    plt.plot(df_test.index, y_test, 'ko-', label='Actual values', linewidth=2, markersize=4)
    # Plot selected model predictions
    for idx in selected_indices:
        if idx < len(Y_PRED_test):
            plt.plot(df_test.index, Y_PRED_test[idx], 
                    label=f'Model {idx+1} ({idx+1} variables)', 
                    linewidth=2, alpha=0.8)
    
    
    
    plt.xlabel('Date', fontweight='bold', fontsize=12)
    plt.ylabel('Number of diagnoses', fontweight='bold', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/selected_models.png")
    if plt_show:
        plt.show()