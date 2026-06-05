"""
Utility Functions for Residual Multivariate Transformers
========================================================

This module contains utility functions for data processing, train-test splitting,
and covariate learning for residual multivariate transformer models.
"""

import pandas as pd
import numpy as np
import sys
import os
import time

from sklearn.preprocessing import MinMaxScaler

from residual_multivariate_transformers.training_evaluation_residual_transformer import load_trained_model

# Add the src directory to path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir)) if 'residual_multivariate_transformers' in current_dir else os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from utils import data_preparation


from src.core.config_manager import get_config

default_config = get_config()


from src.core.data_utils import split_train_test

def filter_diagnostics_covariates(df, diag_codes):
    """
    Filters the DataFrame to keep only the specified diagnostic codes.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input dataframe containing multiple codes
    diag_codes : list
        List of diagnostic codes to retain
        
    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame with only the specified diagnostic codes and timestamp
    """
    if 'timestamp' not in df.columns:
        raise KeyError("Expected a 'timestamp' column in the DataFrame.")
    
    # Ensure diag_codes is a list
    if isinstance(diag_codes, str):
        diag_codes = [diag_codes]
    
    # Filter columns
    cols_to_keep = ['timestamp'] + [code for code in diag_codes if code in df.columns]
    
    if len(cols_to_keep) <= 1:
        raise ValueError("No valid diagnostic codes found in the DataFrame.")
    
    filtered_df = df[cols_to_keep].copy()
    
    print(f"Filtered DataFrame to keep columns: {cols_to_keep}")
    
    return filtered_df

def learn_covariates(df_split, categorical_vars=None):
    """
    Process time series data to extract and prepare covariates.
    
    Parameters:
    -----------
    train_split : pd.DataFrame
        Training data split containing timestamp and target variables
    categorical_vars : list, optional
        List of categorical variables to create (default from config)
        
    Returns:
    --------
    pd.DataFrame
        Processed DataFrame with time series features and covariates
    """
    # Use default categorical variables if not provided
    if categorical_vars is None:
        categorical_vars = default_config.DEFAULT_SEASONAL_CATEGORICAL_VARS.copy()
    
    print(f"Learning covariates with variables: {categorical_vars}")
    
    df = df_split.copy()

    # Call the function from data_preparation module
    df_processed = data_preparation.prepare_time_series_features(df, categorical_vars)

    # Display information about the processed DataFrame
    print("Processed DataFrame info:")
    print(f"  Shape: {df_processed.shape}")
    print(f"  Columns: {list(df_processed.columns)}")
    print(f"  Date range: {df_processed['timestamp'].min()} to {df_processed['timestamp'].max()}")
    
    # Check for any remaining NaN values
    nan_counts = df_processed.isnull().sum()
    if nan_counts.any():
        print("NaN values found in columns:")
        for col, count in nan_counts[nan_counts > 0].items():
            print(f"  {col}: {count}")
    
    return df_processed


def prepare_residual_data(original_predictions, actual_values):
    """
    Compute residuals for residual learning.
    
    Parameters:
    -----------
    original_predictions : np.ndarray
        Predictions from the original model
    actual_values : np.ndarray
        Actual target values
        
    Returns:
    --------
    np.ndarray
        Residuals (actual - predicted)
    """
    residuals = actual_values - original_predictions
    
    print(f"Residual statistics:")
    print(f"  Mean: {np.mean(residuals):.6f}")
    print(f"  Std: {np.std(residuals):.6f}")
    print(f"  Min: {np.min(residuals):.6f}")
    print(f"  Max: {np.max(residuals):.6f}")
    
    return residuals


def create_pandemic_waves_df():
    """
    Create a DataFrame with pandemic wave information.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with pandemic wave periods
    """
    # Convert waves dictionary to DataFrame
    df_waves = pd.DataFrame(default_config.PANDEMIC_WAVES).T.reset_index()
    df_waves.columns = ["Onada", "Inici", "Final"]
    df_waves["Inici"] = pd.to_datetime(df_waves["Inici"])
    df_waves["Final"] = pd.to_datetime(df_waves["Final"])
    
    print("Pandemic waves DataFrame created:")
    print(df_waves.to_string(index=False))
    
    return df_waves


def validate_data_shapes(X, Y, expected_sequence_length=None, expected_features=None):
    """
    Validate the shapes of input data arrays.
    
    Parameters:
    -----------
    X : np.ndarray
        Input features array
    Y : np.ndarray
        Target values array
    expected_sequence_length : int, optional
        Expected sequence length
    expected_features : int, optional
        Expected number of features
        
    Returns:
    --------
    bool
        True if all validations pass
        
    Raises:
    -------
    ValueError
        If validation fails
    """
    print(f"Validating data shapes:")
    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")
    
    # Check that X and Y have the same number of samples
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"Mismatch in number of samples: X has {X.shape[0]}, Y has {Y.shape[0]}")
    
    # Check expected sequence length
    if expected_sequence_length is not None and len(X.shape) > 1:
        if X.shape[1] != expected_sequence_length:
            raise ValueError(f"Expected sequence length {expected_sequence_length}, got {X.shape[1]}")
    
    # Check expected number of features
    if expected_features is not None and len(X.shape) > 2:
        if X.shape[2] != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {X.shape[2]}")
    
    print("✅ Data shape validation passed")
    return True


def load_and_preprocess_data(data_path, target_code=None, split_ratio=None):
    """
    Load and preprocess data from CSV file.
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV data file
    target_code : str, optional
        Target variable code to focus on
    split_ratio : float, optional
        Train-test split ratio (default from config)
        
    Returns:
    --------
    tuple
        (train_df, test_df) - Preprocessed training and testing data
    """
    print(f"Loading data from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load the data
    df = pd.read_csv(data_path)
    print(f"Loaded data shape: {df.shape}")
    
    if target_code:
        print(f"Filtering for target code: {target_code}")
        # Keep only timestamp and target code columns
        if target_code in df.columns and 'timestamp' in df.columns:
            df = df[['timestamp', target_code]].copy()
        else:
            available_cols = list(df.columns)
            raise ValueError(f"Target code '{target_code}' not found. Available columns: {available_cols}")
    
    # Split the data
    train_df, test_df = split_train_test(df, split_ratio=split_ratio)
    
    return train_df, test_df


def extract_model_params_from_filename(model_filename):
    """
    Extract model parameters from filename.
    
    Parameters:
    -----------
    model_filename : str
        Model filename containing parameters
        
    Returns:
    --------
    dict
        Dictionary with extracted parameters (code, forecast, ff_dim, lookback, lr)
    """
    import re
    
    # Pattern to extract parameters from filename
    # Expected format: {code}_*_{forecast}fh_{ff_dim}ff_{lookback}lb_{lr}initlr.keras
    pattern = r'(\w+)_.*_(\d+)fh_(\d+)ff_(\d+)lb_([\d.]+)initlr'
    
    match = re.search(pattern, model_filename)
    if match:
        return {
            'code': match.group(1),
            'forecast': int(match.group(2)),
            'ff_dim': int(match.group(3)),
            'lookback': int(match.group(4)),
            'learning_rate': float(match.group(5))
        }
    else:
        print(f"Warning: Could not extract parameters from filename: {model_filename}")
        return None


def prepare_base_model_data(input_directory, code, lookback, forecast, covid_token=False, cutoff_date='2010-01-01', max_date='2021-06-30', univariate=True, scaler = None, eliminate_covid_data=False, covid_dates=None, split_ratio=None):
    """
    Prepare training and testing data for the base transformer model.
    
    This function loads and prepares univariate time series data for both training and testing.
    It uses the data_preparation module to create input sequences with the specified lookback
    and forecast windows. The function also extracts corresponding date lists for both datasets.
    
    Parameters:
    -----------
    input_directory : str
        Path to the directory containing the input data files
    code : str
        Station or location code identifier for the data
    lookback : int
        Number of past time steps to use as input (lookback window)
    forecast : int
        Number of future time steps to predict (forecast horizon)
    cutoff_date : str, optional
        Date string to filter data from (default is '2010-01-01')
    univariate : bool, optional
        Whether to prepare univariate data (default is True)
        
    Returns:
    --------
    tuple
        A tuple containing (Y_train, Y_test, X_train, X_test, date_list_train, date_list_test):
        - Y_train : np.ndarray, shape (n_train_samples, forecast)
            Training target values
        - Y_test : np.ndarray, shape (n_test_samples, forecast)  
            Testing target values
        - X_train : np.ndarray, shape (n_train_samples, lookback, n_features)
            Training input sequences with univariate features
        - X_test : np.ndarray, shape (n_test_samples, lookback, n_features)
            Testing input sequences with univariate features
        - date_list_train : list
            List of datetime objects corresponding to training samples
        - date_list_test : list  
            List of datetime objects corresponding to testing samples
            
    Notes:
    ------
    - The function uses univariate=True in data_preparation.prepare_data(), which affects
      the number of features in the output arrays
    - Feature count depends on the data_preparation module's univariate processing logic
    - If experiencing shape mismatches, verify that the same univariate setting is used
      during training and inference
    """
    start_time = time.perf_counter()
    X_test, Y_test = data_preparation.prepare_data(
        input_directory, code, lookback, forecast, covid_token=covid_token, cutoff_date=cutoff_date,max_date = max_date,
        train=False, debug=True, univariate=univariate, scaler = scaler, eliminate_covid_data=eliminate_covid_data, 
        covid_dates=covid_dates, split_ratio=split_ratio
    )
    date_list_test = data_preparation.extract_dates(input_directory, code, lookback, forecast, cutoff_date=cutoff_date,max_date = max_date, train=False, eliminate_covid_data=eliminate_covid_data, covid_dates=covid_dates)
    
    X_train, Y_train = data_preparation.prepare_data(
        input_directory, code, lookback, forecast, covid_token=covid_token, cutoff_date=cutoff_date,max_date = max_date,
        train=True, debug=True, univariate=univariate, scaler = scaler, eliminate_covid_data=eliminate_covid_data, covid_dates=covid_dates,
        split_ratio=split_ratio
    )
    date_list_train = data_preparation.extract_dates(input_directory, code, lookback, forecast, cutoff_date=cutoff_date,max_date = max_date, train=True, eliminate_covid_data=eliminate_covid_data, covid_dates=covid_dates)

    finish_preparing = time.perf_counter()
    time_data_preparation = finish_preparing - start_time
    print(f"Data preparation time: {time_data_preparation:.2f} seconds")
    print(f"Test timesteps: {len(date_list_test)}")
    print(f"Train timesteps: {len(date_list_train)}")
    
    

    return Y_train, Y_test, X_train, X_test, date_list_train, date_list_test

def load_base_model_transformer(X_train, X_test,base_path, base_model_name):
    """
    Load a trained base transformer model from the specified path.
    
    Parameters:
    -----------
    base_path : str
        Directory path where the model is stored
    base_model_name : str
        Filename of the model to load
    X_train : np.ndarray
        Training input data for prediction
    X_test : np.ndarray
        Testing input data for prediction
        
    Returns:
    --------
    predictions_train : np.ndarray
        Predictions on the training data
    predictions_test : np.ndarray
        Predictions on the testing data
        
    Raises:
    -------
    FileNotFoundError
        If the model file does not exist
    """
    #Load the base transformer model
    available_models = os.listdir(base_path) if os.path.exists(base_path) else []
    
    if base_model_name in available_models:
        full_path = os.path.join(base_path, base_model_name)
        print(f"✅ Found base model: {base_model_name}")
    else:
        print(f"❌ Base model not found: {base_model_name}")
        print(f"Available models: {available_models}")
        return
    
    # Load the base model
    model = load_trained_model(full_path)
    
    # Get predictions from the base transformer model
    print("\nGenerating predictions from base model...")
    
    predictions_train = model.predict(X_train)
    predictions_test = model.predict(X_test)
    
    return predictions_train, predictions_test