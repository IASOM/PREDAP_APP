import numpy as np
import pandas as pd


from data_utils import data_preparation
from data_utils.data_preparation import split_train_test


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


def prepare_diagnostic_covariate_data(config):
    """Prepare covariate data for residual model"""
    print("\n" + "="*50)
    print("PHASE 2: PREPARING COVARIATE DATA FOR RESIDUAL MODEL")
    print("="*50)
    
    # Load and split the original data for covariate extraction
    train_split, test_split = split_train_test(
        pd.read_csv(config.data_path), 
        split_ratio=0.8, 
        cutoff_date = config.cutoff_date,
        max_date = config.final_cutoff_date,
        #scaler = config.scaler
    )
    
    config.diagnostic_covariates_list = config.load_diagnostic_covariates()
    train_split = config.filter_diagnostics_covariates(train_split, config.diagnostic_covariates_list)
    test_split = config.filter_diagnostics_covariates(test_split, config.diagnostic_covariates_list)
    
    # Generate rolling sequences with covariates for training
    print("Generating sequences with diagnostics covariates for training...")
    config.X_train_covs, _ = data_preparation.prepare_data(
        config.data_path, 
        config.code, 
        config.lookback, 
        config.forecast,
        covid_token=config.covid_token, 
        cutoff_date=config.cutoff_date,
        max_date = config.final_cutoff_date,
        relevant_feature_cols=config.diagnostic_covariates_list, 
        train=True, 
        univariate=False,
        scaler = config.scaler,
        eliminate_covid_data=config.eliminate_covid_data,
        covid_dates=config.covid_dates,
        split_ratio = config.default_split_ratio
    )
    
    print(f"Training covariates shape: {config.X_train_covs.shape}")
    print(f"Expected shape: (num_samples, {config.lookback}, num_features)")
    
    # Generate rolling sequences for test data
    print("Preparing test data with covariates...")
    config.X_test_covs, _ = data_preparation.prepare_data(
        config.data_path, 
        config.code, 
        config.lookback, 
        config.forecast, 
        covid_token=config.covid_token, 
        cutoff_date=config.cutoff_date, 
        max_date = config.final_cutoff_date,
        relevant_feature_cols=config.diagnostic_covariates_list, 
        train=False, 
        univariate=False,
        scaler = config.scaler,
        eliminate_covid_data=config.eliminate_covid_data,
        covid_dates=config.covid_dates,
        split_ratio = config.default_split_ratio
    )
    
    print(f"Test covariates shape: {config.X_test_covs.shape}")
    return config.X_train_covs, config.X_test_covs



def generate_rolling_sequences_covariates(df_processed, lookback, forecast, predictions_train=None, generate_y = False):
    """
    Generates rolling sequences for multivariate time series forecasting.
    
    Parameters:
    - df_processed (pd.DataFrame): Processed DataFrame (should exclude timestamp & target).
    - lookback (int): Number of past timesteps to include in each sequence.
    - forecast (int): Number of future timesteps to predict.
    - predictions_train (np.array, optional): Model predictions to include as a feature. 
      Expected shape: (num_samples, forecast, 1).
    
    Returns:
    - X_train_covs (np.array): Rolling sequences of covariates with shape (samples, forecast, features).
    """

    # Select feature columns (exclude timestamp)
    feature_cols = df_processed.columns[1:]  # Ignore timestamp column
    X_raw = df_processed[feature_cols].values  # Convert DataFrame to NumPy array

    # Generate rolling sequences
    X = [X_raw[i+lookback : i + lookback + forecast] for i in range(len(X_raw) - lookback - forecast + 1)]
    
    
    # Convert to NumPy array
    X_train_covs = np.array(X)

    print(f"Processed covariate data Shapes: X={X_train_covs.shape}")

    # If predictions are provided, concatenate as an extra feature
    if predictions_train is not None:
        # Ensure predictions are correctly shaped
        predictions_train = predictions_train.reshape(X_train_covs.shape[0],forecast, 1)
        X_train_covs = np.concatenate([X_train_covs, predictions_train], axis=-1)  # Add as extra feature
        X_train_covs = X_train_covs.astype(np.float32)
        print(f"Processed covariate Shapes + predictions: X={X_train_covs.shape}")
    
    if generate_y == True:
        Y = [X_raw[i : i + lookback ] for i in range(len(X_raw) - lookback - forecast + 1)]
        Y_train_covs = np.array(Y)
        print(f"Processed covariate data Shapes: Y={Y_train_covs.shape}")
        return Y_train_covs
    else:
        return X_train_covs

def shift_covariates(df, forecast):
    """
    Shifts all covariates back in time by the forecast range to prevent future leakage.

    Parameters:
    - df (pd.DataFrame): DataFrame containing timestamp (first column) and covariates.
    - forecast (int): Number of future timesteps to predict.

    Returns:
    - pd.DataFrame: DataFrame with shifted covariates (NaNs at the start).
    """

    df_shifted = df.copy()

    # Shift all covariate columns (except the timestamp) back by 'forecast' steps
    df_shifted.loc[:,:] = df_shifted.loc[:, :].shift(forecast)

    return df_shifted