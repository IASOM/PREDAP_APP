"""Compatibility wrapper for data preparation utilities.

This module provides the legacy API by delegating to the new `src.data_utils` package.
Keep this wrapper for backward compatibility while modules migrate to `src.data_utils`.
"""
import time

import pandas as pd
import numpy as np
from src.core.data_utils import split_train_test
from src.data_utils.loader import read as read_csv
from src.data_utils.normalizer import normalize_dataframe, inverse_transform_predictions
from src.data_utils.column_mapping import resolve_column, resolve_feature_values
from src.data_utils.features import (
    eliminate_covid_dates,
    add_covid_token,
    prepare_time_series_features,
)
from src.data_utils.sequences import generate_rolling_sequences_covariates, shift_covariates

from pathlib import Path


MAX_DATE = '2025-12-31'


def cut_dataframe(df: pd.DataFrame, date_cutoff: str = "2010-01-01", max_date: str = MAX_DATE, csv_file: str = None, save_data: bool = False) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise KeyError("Expected a 'timestamp' column in the CSV.")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    cutoff = pd.Timestamp(date_cutoff)
    max_dt = pd.Timestamp(max_date)
    df = df[(df["timestamp"] >= cutoff) & (df["timestamp"] <= max_dt)].reset_index(drop=True)

    if save_data and csv_file is not None:
        input_path = Path(csv_file)
        output_file = input_path.parent / f"date_{date_cutoff}_{input_path.name}"
        df.to_csv(output_file, index=False)
        print(f"Cut data saved to: {output_file}")

    return df

def prepare_data(csv_file,code, lookback, forecast, cutoff_date = '2010-01-01', max_date = '2021-06-30', covid_token = False, relevant_feature_cols = None,train = True, debug=False, univariate=True, scaler = None, eliminate_covid_data = False, covid_dates = None, split_ratio=0.8):
    start_time =time.time()
    # Load CSV or Parquet
    df = read_csv(csv_file)
    if eliminate_covid_data:
        assert covid_dates is not None
        df = eliminate_covid_dates(df, covid_dates)
    code = resolve_column(df.columns, code.replace("#", ":"), "target code")
    df = cut_dataframe(df, cutoff_date,max_date, csv_file)
    train_df, test_df = split_train_test(df, split_ratio=split_ratio)
    train_df, test_df = normalize_dataframe(train_df,test_df, csv_file, target_code=code, scaler = scaler)

    if train:
        df = train_df
    else:
        df = test_df

    if univariate:
        # univariate scenario ...................................................
        # Only use the target column as input (Univariate Forecasting)
        #feature_col = df.columns[-1]   # Use the target itself as input
        #target_col = df.columns[-1]    # The future target to predict
        target_col = resolve_column(df.columns, code, "target code")
        feature_cols = target_col

        # Convert to numpy arrays
        X_raw = df[feature_cols].values.reshape(-1, 1)  # Ensure shape is (rows, 1)
        X_raw = X_raw.astype(np.float32)  # Ensure float32 dtype
        Y_raw = df[target_col].values # Target values
        Y_raw = Y_raw.astype(np.float32)  # Ensure float32 dtype
        print(X_raw.shape, Y_raw.shape)
    else: 
        # multivariate scenario ..................................................
        # Select feature columns (exclude timestamp & target)
        target_col = resolve_column(df.columns, code, "target code")
        df_features = df.drop(columns = ['timestamp'])
    
        # Convert DataFrame to numpy arrays
        if relevant_feature_cols is not None:
            X_raw = resolve_feature_values(
                df_features,
                relevant_feature_cols,
                "diagnostic predictor columns",
                fill_missing_with_zero=True,
            )
        else:
            X_raw = df_features.values.astype(np.float32)
        Y_raw = df[target_col].values.astype(np.float32)  # Ensure float32 dtype
    if covid_token:
        df_covid = add_covid_token(df)
        covid_feature = df_covid['covid_token'].values.reshape(-1, 1).astype(np.float32)
        
        X_raw = np.hstack((X_raw, covid_feature))
    # Generate rolling sequences
    X, Y = [], []
    for i in range(len(X_raw) - lookback - forecast + 1):
        X.append(X_raw[i : i + lookback])  # Create `lookback` sequence
        Y.append(Y_raw[i + lookback : i + lookback + forecast])  # Future `forecast` values

    # Convert to numpy arrays
    X, Y = np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

    if debug:
        print(f"Processed Data Shapes: X={X.shape}, Y={Y.shape}")  
    end_time = time.time()
    print(f"Data preparation took {end_time - start_time:.2f} seconds")
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
        X_raw = df_use[feature_cols].values.reshape(-1, 1).astype(np.float32)
        Y_raw = df_use[target_col].values.astype(np.float32)
    else:
        target_col = resolve_column(df_use.columns, code, "target code")
        df_features = df_use.drop(columns=['timestamp'])
        if relevant_feature_cols is not None:
            X_raw = resolve_feature_values(
                df_features,
                relevant_feature_cols,
                "diagnostic predictor columns",
                fill_missing_with_zero=True,
            )
        else:
            X_raw = df_features.values.astype(np.float32)
        Y_raw = df_use[target_col].values.astype(np.float32)

    if covid_token:
        df_covid = add_covid_token(df_use)
        covid_feature = df_covid['covid_token'].values.reshape(-1, 1).astype(np.float32)
        X_raw = __import__('numpy').hstack((X_raw, covid_feature))

    X, Y = [], []
    for i in range(len(X_raw) - lookback - forecast + 1):
        X.append(X_raw[i : i + lookback])
        Y.append(Y_raw[i + lookback : i + lookback + forecast])
    import numpy as _np
    X = _np.array(X, dtype=_np.float32)
    Y = _np.array(Y, dtype=_np.float32)
    if debug:
        print(f"Processed Data Shapes: X={X.shape}, Y={Y.shape}")
    return X, Y


def extract_dates(csv_file, code, lookback, forecast, train=True, cutoff_date='2010-01-01', max_date=MAX_DATE, eliminate_covid_data=False, covid_dates=None, split_ratio=0.8):
    df = read_csv(csv_file)
    if eliminate_covid_data:
        assert covid_dates is not None
        df = eliminate_covid_dates(df, covid_dates)
    if 'timestamp' not in df.columns:
        raise ValueError("The dataset must contain a 'timestamp' column for plotting.")
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    cutoff = pd.Timestamp(cutoff_date)
    max_dt = pd.Timestamp(max_date)
    df = df[(df['timestamp'] >= cutoff) & (df['timestamp'] <= max_dt)].reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_train, df_test = split_train_test(df, split_ratio=split_ratio)
    df_use = df_train if train else df_test
    date_list = df_use['timestamp'].iloc[lookback : len(df_use) - forecast + 1].reset_index(drop=True)
    return date_list.tolist()
