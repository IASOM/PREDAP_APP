# visualization_lmlr.py
# ------------------------------------------------------
# Visualization utilities for LMLR models
# Author: Guillem Hernández Guillamet
# Version: 1.0
# Date: 04/06/2025
# ------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_example(df, title, show_plt = False):
    """
    Plot a sample of time series from the DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Time series data
    title : str
        Plot title
    """
    sampled_cols = np.random.choice(df.columns, size=min(10, len(df.columns)), replace=False)
    dff = df[sampled_cols].copy()
    dff["date"] = dff.index
    sns.set_theme(rc={'figure.figsize': (20, 8)})
    ax = sns.lineplot(data=dff.melt(id_vars=['date']), x='date', y='value', hue='variable')
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_xlabel('Date', fontweight='bold', fontsize=12)
    ax.set_ylabel('Value', fontweight='bold', fontsize=12)
    plt.legend(title='Variables', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/example_plot_{title}.png")
    if show_plt:
        plt.show()
    plt.close()


def ploter(df, title, n, code, show_plt = False):
    """
    Plot top n variables plus the target variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Time series data
    title : str
        Plot title
    n : int
        Number of top variables to plot
    code : str
        Target variable name
    """
    cols = list(df.columns)[:n] + [code]
    dff = df[cols]
    dff["date"] = dff.index
    sns.set_theme(rc={'figure.figsize': (20, 8)})
    ax = sns.lineplot(data=dff.melt(id_vars=['date']), x='date', y='value', hue='variable')
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_xlabel('Date', fontweight='bold', fontsize=12)
    ax.set_ylabel('Value', fontweight='bold', fontsize=12)
    plt.legend(title='Variables', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if show_plt:
        plt.show()
    plt.close()


def plot_models(df_test, code, corr, Y_PREDICTIONS_test, y_test, show_plt = False):
    """
    Plot predictions from all models vs actual values.
    
    Parameters:
    -----------
    df_test : pd.DataFrame
        Test data with datetime index
    code : str
        Target variable name
    corr : pd.Series
        Correlation series (used for model count)
    Y_PREDICTIONS_test : list
        List of predictions for each model
    y_test : pd.Series
        Actual test values
    """
    fig, ax = plt.subplots(figsize=(20, 15))
    fig.suptitle(f'{code} prediction', fontweight='bold', fontsize=14)
    # Plot actual values with emphasis
    ax.plot(y_test.index, y_test, 'go-', 
           label='Actual values', linewidth=1, markersize=4)
    # Plot each model prediction
    for i in range(len(corr)-1):
        ax.plot(y_test.index, Y_PREDICTIONS_test[i], 
               label=f'Model {i+1} ({i+1} vars)', linewidth=1, alpha=0.8)
    

    # Set bold axis labels
    ax.set_xlabel('Date', fontweight='bold', fontsize=12)
    ax.set_ylabel('n diagnoses', fontweight='bold', fontsize=12)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/MLR_plot_{code}.png")
    if show_plt:
        plt.show()
    plt.close()


def plot_metrics(df, code, show_plt = False):
    """
    Plot model performance metrics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing model metrics
    """
    # F-statistics plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.lineplot(data=df[['number_of_variables', "F1", "F2"]].melt(id_vars=['number_of_variables']),
                x='number_of_variables', y='value', hue="variable")
    plt.title('F-Statistics vs Number of Variables', fontweight='bold')
    plt.xlabel('Number of Variables', fontweight='bold')
    plt.ylabel('F-Statistic', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # P-values plot
    plt.subplot(1, 2, 2)
    sns.lineplot(data=df[['number_of_variables', "pval1", "pval2"]].melt(id_vars=['number_of_variables']),
                x='number_of_variables', y='value', hue="variable")
    plt.title('P-Values vs Number of Variables', fontweight='bold') 
    plt.xlabel('Number of Variables', fontweight='bold')
    plt.ylabel('P-Value', fontweight='bold')

    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"plots/metrics_plot_{code}.png")
    if show_plt:
        plt.show()
    plt.close()