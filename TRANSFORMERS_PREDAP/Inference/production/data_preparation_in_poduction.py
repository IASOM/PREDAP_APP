"""Data preparation helpers for production inference."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import FunctionTransformer

from config.base_transformer_config import BaseTransformerConfig
from data_utils import data_preparation


default_config = BaseTransformerConfig()


class DataPreparationInProduction:
    def __init__(self, config: BaseTransformerConfig = default_config):
        self.config = config
        self.config.print_config()

    def load_diagnostic_covariates(self, diagnostic_covariates_path, code, forecast):
        diagnostic_covariates_df = pd.read_excel(diagnostic_covariates_path + code + ".xlsx", engine="openpyxl")
        diagnostic_covariates_list = list(diagnostic_covariates_df[diagnostic_covariates_df["LAG"] == forecast]["predictors"])[0].split(",")
        return diagnostic_covariates_list

    def prepare_prediction_univ_data(
        self,
        data_path: str,
        code: str,
        lookback: int,
        forecast: int,
        cutoff_date: str,
        max_date: str,
        scaler: FunctionTransformer,
        eliminate_covid_data: bool,
        covid_token: bool,
        production_mode: bool = False,
        covid_dates: Optional[List[str]] = None,
    ):
        code = code.replace("#", ":")
        df = pd.read_csv(data_path)
        if eliminate_covid_data:
            if covid_dates is None:
                raise ValueError("covid_dates must be provided when eliminate_covid_data=True")
            df = data_preparation.eliminate_covid_dates(df, covid_dates)
        df = data_preparation.cut_dataframe(df, cutoff_date, max_date, data_path)
        categorical_vars = ["Day_of_Week", "Month", "Season", "Holiday", "School_Vacation", "Is_Weekend"]
        df_dates = data_preparation.prepare_time_series_features(
            df,
            categorical_vars=categorical_vars,
            cutoff_date=cutoff_date,
            max_date=max_date,
            scaler=scaler,
            eliminate_covid_data=eliminate_covid_data,
            covid_dates=covid_dates,
        )
        df_timestamp = df["timestamp"]
        df_dates = df_dates.drop(columns=["timestamp"])

        code = "DEMAND_" + code if not code.startswith("DEMAND_") else code
        code = code.replace("__", "_") if "__" in code else code
        idx_code = df.columns.get_loc(code)
        feature_cols = df.columns[idx_code]
        target_col = df.columns[idx_code]

        X_raw = df[feature_cols].values.reshape(-1, 1)
        X_raw = np.hstack((X_raw, df_dates.values.astype(np.float32)))
        Y_raw = df[target_col].values

        if covid_token:
            df_covid = data_preparation.add_covid_token(df)
            covid_feature = df_covid["covid_token"].values.reshape(-1, 1)
            X_raw = np.hstack((X_raw, covid_feature))

        if production_mode:
            X_raw = X_raw[-lookback:].reshape(1, lookback, -1)
            Y_raw = np.zeros_like(Y_raw[-forecast:].reshape(1, forecast))
            last_date = df_timestamp.iloc[-1]
            new_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast, freq="D")
            df_forecast = pd.Series(new_dates)
            df_timestamp = pd.concat([df_timestamp, df_forecast], ignore_index=True)
        else:
            X_raw = X_raw[-(lookback + forecast) : -forecast].reshape(1, lookback, -1)
            Y_raw = Y_raw[-forecast:].reshape(1, forecast)

        return X_raw, Y_raw, df_timestamp

    def prepare_prediction_diagnostics_data(
        self,
        data_path: str,
        code: str,
        lookback: int,
        forecast: int,
        cutoff_date: str,
        max_date: str,
        scaler: FunctionTransformer,
        covid_token: bool = False,
        production_mode: bool = False,
        covid_dates: Optional[List[str]] = None,
        eliminate_covid_data: bool = False,
    ):
        relevant_feature_cols = self.load_diagnostic_covariates(default_config.diagnostic_covariates_path, code, forecast)
        code = code.replace("#", ":")
        df = pd.read_csv(data_path)
        if eliminate_covid_data:
            if covid_dates is None:
                raise ValueError("covid_dates must be provided when eliminate_covid_data=True")
            df = data_preparation.eliminate_covid_dates(df, covid_dates)
        df = data_preparation.cut_dataframe(df, cutoff_date, max_date, data_path)

        categorical_vars = ["Day_of_Week", "Month", "Season", "Holiday", "School_Vacation", "Is_Weekend"]
        df_dates = data_preparation.prepare_time_series_features(
            df,
            categorical_vars=categorical_vars,
            cutoff_date=cutoff_date,
            max_date=max_date,
            scaler=scaler,
            eliminate_covid_data=eliminate_covid_data,
            covid_dates=covid_dates,
        )
        df_dates = df_dates.drop(columns=["timestamp"])

        old_code = code
        code = "DEMAND_" + code if not code.startswith("DEMAND_") else code
        code = code.replace("__", "_") if "__" in code else code

        df.columns = [col if col == "timestamp" else ("DEMAND_" + col if not col.startswith("DEMAND_") else col).replace("__", "_") for col in df.columns]
        if code not in df.columns:
            code = old_code

        idx_code = df.columns.get_loc(code)
        df_features = df.drop(columns=["timestamp"])
        target_col = df.columns[idx_code]

        if relevant_feature_cols is not None:
            X_raw = df_features[relevant_feature_cols].values
            X_raw = np.hstack((X_raw, df_dates.values.astype(np.float32)))
        else:
            X_raw = df_features.values

        Y_raw = df[target_col].values
        if covid_token:
            df_covid = data_preparation.add_covid_token(df)
            covid_feature = df_covid["covid_token"].values.reshape(-1, 1)
            X_raw = np.hstack((X_raw, covid_feature))

        if production_mode:
            X_raw = X_raw[-lookback:].reshape(1, lookback, -1)
            Y_raw = np.zeros_like(Y_raw[-forecast:].reshape(1, forecast))
        else:
            X_raw = X_raw[-(lookback + forecast) : -forecast].reshape(1, lookback, -1)
            Y_raw = Y_raw[-forecast:].reshape(1, forecast)
        return X_raw, Y_raw

    def prepare_prediction_seasonal_data(
        self,
        data_path: str,
        code: str,
        forecast: int,
        lookback: int,
        cutoff_date: str,
        max_date: str,
        categorical_vars: List[str],
        predictions_train: Optional[np.ndarray],
        predictions_test: Optional[np.ndarray],
        scaler: FunctionTransformer,
    ):
        df = pd.read_csv(data_path)
        print("Preparing seasonal features for training data...")
        df_processed = data_preparation.prepare_time_series_features(
            df,
            self.config.DEFAULT_SEASONAL_CATEGORICAL_VARS,
            cutoff_date=self.config.cutoff_date,
            max_date=self.config.final_cutoff_date,
            scaler=self.config.scaler,
            eliminate_covid_data=self.config.eliminate_covid_data,
            covid_dates=self.config.covid_dates,
        )
        X_seasonal_covs = df_processed.drop(columns=["timestamp"]).values[-forecast:].reshape(1, -1, df_processed.shape[1] - 1)
        return X_seasonal_covs.astype(float)
