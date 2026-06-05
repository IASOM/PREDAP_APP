import numpy as np

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
