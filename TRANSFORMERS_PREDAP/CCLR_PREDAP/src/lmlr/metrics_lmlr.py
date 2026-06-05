# metrics_lmlr.py
# ------------------------------------------------------
# Evaluation metrics and performance analysis utilities
# Author: Guillem Hernández Guillamet
# Version: 1.0
# Date: 04/06/2025
# ------------------------------------------------------

import pandas as pd
import numpy as np
import scipy.stats
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error


def metrics_calculation(models, y_train, y_test, Y_PRED_train, Y_PRED_test, params):
    """
    Calculate performance metrics for a series of nested models.
    
    Parameters:
    -----------
    models : list
        List of trained statsmodels OLS models
    y_train : pd.Series
        Training target values
    y_test : pd.Series
        Test target values
    Y_PRED_train : list
        Training predictions for each model
    Y_PRED_test : list
        Test predictions for each model
    params : list
        Parameter descriptions for each model
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing metrics for each model comparison
    """
    results = []
    
    for i in range(len(models) - 1):
        n = len(y_train)
        k1 = len(models[i].params)
        k2 = len(models[i+1].params)
        rss1, rss2 = models[i].ssr, models[i+1].ssr
        
        # F-test 1: Nested model comparison (standard)
        Fstat1 = ((rss1 - rss2)/(k2 - k1)) / (rss2/(n - k2))
        #pval1 = 1 - scipy.stats.f.cdf(Fstat1, k2 - k1, n - k2)
        pval1 = 1 - scipy.stats.f.cdf(Fstat1, k2 - k1, n - k2- 1)
        
        # F-test 2: Alternative formulation (keep for compatibility)
        #Fstat2 = ((Y_PRED_train[i+1] - y_train.mean()).pow(2).sum() -
        #          (Y_PRED_train[i] - y_train.mean()).pow(2).sum()) / (k2 - k1) / (sum((Y_PRED_train[i+1] - y_train)**2) / (n - k2 - 1))
        #pval2 = 1 - scipy.stats.f.cdf(Fstat2, k2 - k1, n - k2 - 1)
        
        sse2 = np.sum((Y_PRED_train[i+1].values - y_train)**2)
        ssr1 = np.sum((Y_PRED_train[i].values - y_train.mean())**2)
        ssr2 = np.sum((Y_PRED_train[i+1].values - y_train.mean())**2)

        Fstat2 = ((ssr2-ssr1)/(k2-k1))/(sse2/(n-k2-1))
        pval2 = 1-scipy.stats.f.cdf(Fstat2, k2-k1, n-k2-1)

        results.append({
            'number_of_variables': i + 1,  # Fixed: should be i+1 for first model
            'F1': Fstat1,
            'pval1': pval1,
            'F2': Fstat2,
            'pval2': pval2,
            'MAPE_train': mean_absolute_error(y_train, Y_PRED_train[i]) * 100,
            'MAPE_test': mean_absolute_error(y_test, Y_PRED_test[i]) * 100,
            'RMSE_train': math.sqrt(mean_squared_error(y_train, Y_PRED_train[i])),
            'RMSE_test': math.sqrt(mean_squared_error(y_test, Y_PRED_test[i])),
            'predictors': params[i]
        })
    
    return pd.DataFrame(results)


def evaluation_metrics_MAPE(BEST, code, plt_show = False):
    """
    Plot MAPE analysis for lagged predictions.
    
    Parameters:
    -----------
    BEST : pd.DataFrame
        DataFrame containing best model results with LAG column
    """
    dff = BEST[["MAPE_train", "MAPE_test", "LAG"]].copy()
    
    plt.figure(figsize=(20, 8))
    sns.lineplot(data=dff.replace('nan', float('nan')).melt(id_vars=['LAG']), x='LAG', y='value', hue='variable')
    plt.title("MAPE Analysis in Lagged Prediction", fontweight='bold', fontsize=14)
    
    # Add vertical lines for key lag periods
    max_val = BEST[["MAPE_train", "MAPE_test"]].max(0).max(0)
    for lag in [7, 14, 21, 28]:
        plt.axvline(lag, 0, max_val, color="red", alpha=0.7, linestyle='--')
        plt.text(lag, max_val * 0.9, f'{lag}d', rotation=90, 
                verticalalignment='top', fontweight='bold')
    plt.xlabel('Lag (days)', fontweight='bold', fontsize=12)
    plt.ylabel('MAPE (%)', fontweight='bold', fontsize=12)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/MAPE_analysis_{code}.png")
    if plt_show:
        plt.show()
    plt.close()


def evaluation_metrics_RMSE(BEST, code, plt_show = False):
    """
    Plot RMSE analysis for lagged predictions.
    
    Parameters:
    -----------
    BEST : pd.DataFrame
        DataFrame containing best model results with LAG column
    """
    dff = BEST[["RMSE_train", "RMSE_test", "LAG"]].copy()
    
    plt.figure(figsize=(20, 8))
    sns.lineplot(data=dff.replace('nan', float('nan')).melt(id_vars=['LAG']),
                x='LAG', y='value', hue='variable')
    plt.title("RMSE Analysis in Lagged Prediction", fontweight='bold', fontsize=14)
    plt.xlabel('Lag (days)', fontweight='bold', fontsize=12)
    plt.ylabel('RMSE', fontweight='bold', fontsize=12)
    
    # Add vertical lines for key lag periods
    max_val = BEST[["RMSE_train", "RMSE_test"]].max().max()
    for lag in [7, 14, 21, 28]:
        plt.axvline(lag, 0, max_val, color="red", alpha=0.7, linestyle='--')
        plt.text(lag, max_val * 0.9, f'{lag}d', rotation=90, 
                verticalalignment='top', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/RMSE_analysis_{code}.png")
    if plt_show:
        plt.show()
    plt.close()


def evaluation_metrics_Ftest(BEST, code, plt_show = False):
    """
    Plot F-test analysis for lagged predictions.
    
    Parameters:
    -----------
    BEST : pd.DataFrame
        DataFrame containing best model results with LAG column
    """
    required_cols = ["F1", "pval1", "F2", "pval2", "LAG"]
    
    # Check if all required columns exist
    missing_cols = [col for col in required_cols if col not in BEST.columns]
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}. Skipping F-test plot.")
        return
    
    dff = BEST[required_cols].copy()
    
    plt.figure(figsize=(20, 8))
    sns.lineplot(data=dff.replace('nan', float('nan')).melt(id_vars=['LAG']),
                x='LAG', y='value', hue='variable')
    plt.title("F-Statistics Analysis in Lagged Prediction", fontweight='bold', fontsize=14)
    plt.xlabel('Lag (days)', fontweight='bold', fontsize=12)
    plt.ylabel('F-Statistic / P-Value', fontweight='bold', fontsize=12)
    
    # Add vertical lines for key lag periods
    max_val = BEST[["F1", "pval1", "F2", "pval2"]].max().max()
    for lag in [7, 14, 21, 28]:
        plt.axvline(lag, 0, max_val, color="red", alpha=0.7, linestyle='--')
        plt.text(lag, max_val * 0.9, f'{lag}d', rotation=90, 
                verticalalignment='top', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for better visualization of p-values
    plt.tight_layout()
    plt.savefig(f"plots/Ftest_analysis_{code}.png")
    if plt_show:
        plt.show()
    plt.close()