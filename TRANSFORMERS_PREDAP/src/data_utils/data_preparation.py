"""Compatibility wrapper for data preparation utilities.

This module provides the legacy API by delegating to the new `src.data_utils` package.
Keep this wrapper for backward compatibility while modules migrate to `src.data_utils`.
"""
import pandas as pd
from src.data_utils.features import (
    cut_dataframe
)
from src.data_utils.features import (
    cut_dataframe,
    prepare_time_series_features,
)

#from src.data_utils.loader import read as read_csv
from src.utils.experiments_utils import smart_read as read_csv
from src.data_utils.normalizer import normalize_dataframe, inverse_transform_predictions
from src.data_utils.column_mapping import resolve_column, resolve_feature_values
from src.data_utils.features import (
    eliminate_covid_dates,
    add_covid_token,
)

from typing import Tuple, Optional

import time

import numpy as np


MAX_DATE = '2025-12-31'

def generate_rolling_sequences(X_raw, Y_raw, lookback, forecast):
    # Generate rolling sequences
    X, Y = [], []
    for i in range(len(X_raw) - lookback - forecast + 1):
        X.append(X_raw[i : i + lookback])  # Create `lookback` sequence
        Y.append(Y_raw[i + lookback : i + lookback + forecast])  # Future `forecast` values

    # Convert to numpy arrays
    X, Y = np.array(X), np.array(Y)
    return X, Y


def prepare_univariate_data(df ,code, lookback, forecast, cutoff_date = '2010-01-01', max_date = '2026-12-31', covid_token = False, scaler = None, eliminate_covid_data = False, covid_dates = None):

    if eliminate_covid_data:
        assert covid_dates is not None
        df = eliminate_covid_dates(df, covid_dates)
    code = resolve_column(df.columns, code.replace("#", ":"), "target code")
    df = cut_dataframe(df, cutoff_date,max_date)
    df = normalize_dataframe(df, target_code=code, scaler = scaler)

    # univariate scenario ...................................................
    # Only use the target column as input (Univariate Forecasting)
    target_col = resolve_column(df.columns, code, "target code")
    feature_cols = target_col

    # Convert to numpy arrays
    X_raw = df[feature_cols].values.reshape(-1, 1)  # Ensure shape is (rows, 1)
    Y_raw = df[target_col].values # Target values

    if covid_token:
        df_covid = add_covid_token(df)
        covid_feature = df_covid['covid_token'].values.reshape(-1, 1)
        X_raw = np.hstack((X_raw, covid_feature))

    # Generate rolling sequences
    X, Y = generate_rolling_sequences(X_raw, Y_raw, lookback, forecast)

    return X, Y


def prepare_multivariate_data(df ,code, lookback, forecast, cutoff_date = '2010-01-01', max_date = '2021-06-30', covid_token = False, relevant_feature_cols = None, scaler = None, eliminate_covid_data = False, covid_dates = None):

    if eliminate_covid_data:
        assert covid_dates is not None
        df = eliminate_covid_dates(df, covid_dates)
    code = resolve_column(df.columns, code.replace("#", ":"), "target code")
    df = cut_dataframe(df, cutoff_date,max_date)
    df = normalize_dataframe(df, target_code=code, scaler = scaler)

    # multivariate scenario ..................................................
    # Select feature columns (exclude timestamp & target)
    df_features = df.drop(columns = ['timestamp'])
    target_col = resolve_column(df.columns, code, "target code")

    # Convert DataFrame to numpy arrays
    if relevant_feature_cols is not None:
        X_raw = resolve_feature_values(
            df_features,
            relevant_feature_cols,
            "diagnostic predictor columns",
            fill_missing_with_zero=True,
        )
    else:
        X_raw = df_features.values
    Y_raw = df[target_col].values

    if covid_token:
        df_covid = add_covid_token(df)
        covid_feature = df_covid['covid_token'].values.reshape(-1, 1)
        X_raw = np.hstack((X_raw, covid_feature))
        
    # Generate rolling sequences
    X, Y = generate_rolling_sequences(X_raw, Y_raw, lookback, forecast)  

    return X, Y


def prepare_data_not_normalized(csv_file, code, lookback, forecast, cutoff_date='2010-01-01', max_date='2025-09-30', covid_token=False, relevant_feature_cols=None, train=True, debug=False, univariate=True, eliminate_covid_data=False, covid_dates=None, split_ratio=0.8):
    df = read_csv(csv_file)
    if eliminate_covid_data:
        assert covid_dates is not None
        df = eliminate_covid_dates(df, covid_dates)
    code = resolve_column(df.columns, code.replace("#", ":"), "target code")
    df = df[(df['timestamp'] >= cutoff_date) & (df['timestamp'] <= max_date)].reset_index(drop=True)
    train_df, test_df = split_train_test(df, split_ratio=split_ratio)
    df_use = train_df if train else test_df

    if univariate:
        target_col = resolve_column(df_use.columns, code, "target code")
        feature_cols = target_col
        X_raw = df_use[feature_cols].values.reshape(-1, 1)
        Y_raw = df_use[target_col].values
    else:
        df_features = df_use.drop(columns=['timestamp'])
        target_col = resolve_column(df_use.columns, code, "target code")
        if relevant_feature_cols is not None:
            X_raw = resolve_feature_values(
                df_features,
                relevant_feature_cols,
                "diagnostic predictor columns",
                fill_missing_with_zero=True,
            )
        else:
            X_raw = df_features.values
        Y_raw = df_use[target_col].values

    if covid_token:
        df_covid = add_covid_token(df_use)
        covid_feature = df_covid['covid_token'].values.reshape(-1, 1)
        X_raw = __import__('numpy').hstack((X_raw, covid_feature))

    X, Y = generate_rolling_sequences(X_raw, Y_raw, lookback, forecast)

    import numpy as _np
    X = _np.array(X).astype(_np.float32)
    Y = _np.array(Y).astype(_np.float32)
    if debug:
        print(f"Processed Data Shapes: X={X.shape}, Y={Y.shape}")
    return X, Y


def extract_dates(csv_file, code, lookback, forecast, train=True, cutoff_date='2010-01-01', max_date=MAX_DATE, eliminate_covid_data=False, covid_dates=None):
    """
    Extract corresponding date list for the given data preparation parameters.
    """
    
    cutoff = pd.Timestamp(cutoff_date)
    max_dt = pd.Timestamp(max_date)

    df = read_csv(csv_file)

    if eliminate_covid_data:
        assert covid_dates is not None
        df = eliminate_covid_dates(df, covid_dates)
    if 'timestamp' not in df.columns:
        raise ValueError("The dataset must contain a 'timestamp' column for plotting.")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df[(df['timestamp'] >= cutoff) & (df['timestamp'] <= max_dt)].reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df_train, df_test = split_train_test(df)
    df_use = df_train if train else df_test
    date_list = df_use['timestamp'].iloc[lookback : len(df_use) - forecast + 1].reset_index(drop=True)
    return date_list.tolist()

def split_train_test(df: pd.DataFrame,
                     split_ratio: float = 0.8,
                     cutoff_date: Optional[str] = None,
                     max_date: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Unified train/test split utility.
    Filters by cutoff_date/max_date if provided and returns train/test split.
    """
    df = df.copy()
    if cutoff_date:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df[df['timestamp'] >= pd.to_datetime(cutoff_date)]
    if max_date:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df[df['timestamp'] <= pd.to_datetime(max_date)]
    n = int(len(df) * split_ratio)
    train = df.iloc[:n].reset_index(drop=True)
    test = df.iloc[n:].reset_index(drop=True)
    return train, test

