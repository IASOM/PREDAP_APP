# visualization_dl.py
# ====================
# Visualization utilities for deep learning models
# Author: Guillem Hernández Guillamet
# Version: 1.0

import matplotlib.pyplot as plt
import pandas as pd


def plot_train_test(train_dataset, test_dataset, objective, show_plt = False):
    """
    Plot training and test datasets for visual inspection.
    
    Parameters:
    -----------
    train_dataset : pd.DataFrame
        Training dataset
    test_dataset : pd.DataFrame
        Test dataset
    objective : str
        Name of the target variable column
        
    Example:
    --------
    >>> plot_train_test(train_df, test_df, 'timeseries_350')
    """
    try:
        plt.figure(figsize=(10, 4))
        plt.plot(train_dataset[objective])
        plt.plot(test_dataset[objective])
        plt.xlabel('Time (day)')
        plt.ylabel('n diags Hipertension')
        plt.legend(['Train set', 'Test set'], loc='upper right')
        plt.savefig(f"plots/train_test_split_{objective}.png")
        if show_plt:
            plt.show()
        plt.close()

        print('Dimension of train data:', train_dataset.shape)
        print('Dimension of test data:', test_dataset.shape)
        
    except Exception as e:
        print(f"Error plotting train/test data: {str(e)}")


def plt_model(y_test_inverse, yhat_inverse, model_name,objective = None, col_idx=None, show_plt = False):
    """
    Plot model results comparing true vs predicted values.
    
    Parameters:
    -----------
    y_test_inverse : np.ndarray
        True values (inverse transformed)
    yhat_inverse : np.ndarray
        Predicted values (inverse transformed)
    model_name : str
        Name of the model for the plot title
    col_idx : int, optional
        Column index to plot (defaults to global col_idx)
        
    Example:
    --------
    >>> plt_model(y_true, y_pred, "LSTM", col_idx=0)
    """
    try:
        # Use global col_idx if not provided
        if col_idx is None:
            col_idx = globals().get('col_idx', 0)
            
        plt.figure(figsize=(20, 10))
        plt.plot(pd.DataFrame(y_test_inverse)[[col_idx]], label='True Values')
        plt.plot(pd.DataFrame(yhat_inverse)[[col_idx]], label='Predicted Values')
        plt.set_xlabel('Date', fontweight='bold', fontsize=12)
        plt.set_ylabel('Value', fontweight='bold', fontsize=12)
        plt.title(f'Real_{objective} vs. Predicted Values_{objective} // MODEL: {model_name}')
        plt.legend()
        plt.savefig(f"CCLR_PREDAP/plots/model_results_{model_name}_{objective}.png")
        if show_plt:
            plt.show()
        plt.close()

    except Exception as e:
        print(f"Error plotting model results: {str(e)}")