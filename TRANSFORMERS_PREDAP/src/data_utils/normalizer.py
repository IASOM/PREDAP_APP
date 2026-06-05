from typing import Any
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import clone
import numpy as np
import pandas as pd


def normalize_dataframe(df: pd.DataFrame, target_code: str = None, scaler: Any =None):
    if 'timestamp' not in df.columns:
        raise KeyError("Expected a 'timestamp' column in the CSV.")
    df['timestamp'] = df['timestamp'].astype('datetime64[ns]')

    if scaler is None:
        scaler = MinMaxScaler()
        scaler_target = MinMaxScaler()
    else:
        scaler_target = clone(scaler)

    codes = [code for code in df.columns if (code != 'timestamp' and code != target_code)]
    scaler.fit(df[codes].values)
    df[codes] = scaler.transform(df[codes].values)
    if target_code is not None:
        scaler_target = scaler_target.fit(df[[target_code]].values)
        df[[target_code]] = scaler_target.transform(df[[target_code]].values)

    return df


def inverse_transform_predictions(predictions, original_scale_df, code, lookback, forecast, cutoff_date='2010-01-01', max_date='2025-09-30', scaler = None, eliminate_covid_data=False, covid_dates=None, split_ratio = 0.8):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.base import clone
    code = code.replace("#", ":")
    if eliminate_covid_data:
        assert covid_dates is not None
        # Filter out covid dates
        for start_date, end_date in covid_dates:
            original_scale_df['timestamp'] = original_scale_df['timestamp'].astype('datetime64[ns]')
            original_scale_df = original_scale_df[~((original_scale_df['timestamp'] >= start_date) & (original_scale_df['timestamp'] <= end_date))]

    original_scale_df['timestamp'] = original_scale_df['timestamp'].astype('datetime64[ns]')
    original_scale_df = original_scale_df[(original_scale_df['timestamp'] >= cutoff_date) & (original_scale_df['timestamp'] <= max_date)].reset_index(drop=True)

    n = int(len(original_scale_df) * split_ratio)
    train_df = original_scale_df.iloc[:n]

    train_df_seq = []
    for i in range(len(train_df[[code]]) - lookback - forecast + 1):
        train_df_seq.append(train_df[[code]].values[i + lookback : i + lookback + forecast])
    train_df_seq = np.array(train_df_seq).squeeze()

    if scaler is None:
        scaler_target = MinMaxScaler()
    else:
        scaler_target = clone(scaler)

    if len(train_df_seq.shape) < 2:
        train_df_seq = train_df_seq.reshape(-1,1)

    scaler_target.fit(train_df_seq)
    pred_original_scale = scaler_target.inverse_transform(predictions)
    return pred_original_scale
