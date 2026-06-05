"""Prediction and reconstruction pipeline used for inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.pipeline import FunctionTransformer

from config.base_transformer_config import BaseTransformerConfig
from production.data_preparation_in_poduction import DataPreparationInProduction
from model_architechture import model_architecture_residual_transformer
from model_architechture import model_architecture_univ_transformer
from utils.experiments_utils import smart_read
from data_utils import data_preparation

pd.read_csv = smart_read


default_config = BaseTransformerConfig()


class ModelPredictionPipeline(DataPreparationInProduction):
    def __init__(self, config: BaseTransformerConfig):
        self.config = config
        self.config.print_config()

    def create_univariate_transformer_model(self, input_shape, forecast, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, activation_function="tanh", dropout=0, mlp_dropout=0, n_pred=1, pos_encoding=True):
        return model_architecture_univ_transformer.build_model(
            input_shape=input_shape,
            head_size=head_size,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_transformer_blocks=num_transformer_blocks,
            mlp_units=mlp_units,
            activation_function=activation_function,
            dropout=dropout,
            mlp_dropout=mlp_dropout,
            n_pred=forecast,
            pos_encoding=pos_encoding,
        )

    def create_residual_transformer_model(self, input_shape, forecast, transformer_params, activation_function="tanh"):
        return model_architecture_residual_transformer.hybrid_lstm_transformer_model(
            input_shape=input_shape,
            forecast=forecast,
            transformer_params=transformer_params,
            activation_function=activation_function,
        )

    def load_model_weights(self, model, code, lookback, forecast, models_directory, model_type="univariate"):
        weights_path = f"{models_directory}/{code}/{model_type}/{code}_{model_type}_{forecast}fh_{lookback}lb_f16_weights.h5"
        print(model.summary())
        try:
            model.load_weights(weights_path)
            print(f"Successfully loaded weights from {weights_path}")
        except Exception as error:
            print(f"Error loading weights: {error}")
            raise
        return model

    def reconstruct_full_model(self, code: str, lookback: int, forecast: int, models_directory: str, univ_input_shape: Tuple[int], diagnostics_input_shape: Tuple[int], seasonal_input_shape: Tuple[int], head_size: int, num_heads: int, ff_dim: int, num_transformer_blocks: int, mlp_units: List[int], activation_function: str = "tanh", dropout: float = 0, mlp_dropout: float = 0, n_pred: int = 1, pos_encoding=True):
        default_residual_transformer_params = {
            "head_size": 16,
            "num_heads": 16,
            "ff_dim": 512,
            "mlp_units": [256, 128],
            "num_transformer_blocks": 2,
            "dropout": 0,
        }

        univ_model = self.create_univariate_transformer_model(
            input_shape=univ_input_shape,
            forecast=forecast,
            head_size=head_size,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_transformer_blocks=num_transformer_blocks,
            mlp_units=mlp_units,
            activation_function=activation_function,
            dropout=dropout,
            mlp_dropout=mlp_dropout,
            n_pred=forecast,
            pos_encoding=pos_encoding,
        )

        diagnostics_model = self.create_residual_transformer_model(
            input_shape=diagnostics_input_shape,
            forecast=forecast,
            transformer_params=default_residual_transformer_params,
            activation_function=activation_function,
        )

        seasonal_model = self.create_residual_transformer_model(
            input_shape=seasonal_input_shape,
            forecast=forecast,
            transformer_params=default_residual_transformer_params,
            activation_function=activation_function,
        )

        def process_col_name(col):
            if col == "timestamp":
                return col
            code_name = col[7:] if col.startswith("DEMAND_") else col
            indices = [i for i, char in enumerate(code_name) if char == "_"]
            if not indices:
                return code_name
            if len(indices) == 1:
                idx = indices[0]
                return code_name[:idx] + "__" + code_name[idx + 1 :]
            first_idx = indices[0]
            last_idx = indices[-1]
            return code_name[:first_idx] + "__" + code_name[first_idx + 1 : last_idx] + "__" + code_name[last_idx + 1 :]

        code = code[7:] if code.startswith("DEMAND_") else code
        code = process_col_name(code)

        univ_model = self.load_model_weights(univ_model, code, lookback, forecast, models_directory, model_type="univariate_model")
        diagnostics_model = self.load_model_weights(diagnostics_model, code, lookback, forecast, models_directory, model_type="diagnostics_model")
        seasonal_model = self.load_model_weights(seasonal_model, code, lookback, forecast, models_directory, model_type="seasonal_model")
        return univ_model, diagnostics_model, seasonal_model

    def compute_predap_auxiliary_metrics(self, predictions: np.ndarray, true_past_data: np.ndarray, mae: np.ndarray, lookback: int, confidence_level: float = 0.95) -> Dict[str, float]:
        pred_series = pd.Series(predictions.flatten())
        true_past_data = pd.Series(true_past_data.flatten())
        full_series = pd.concat([true_past_data, pred_series], ignore_index=True)
        k_multiplier = -np.log(2 * (1.0 - confidence_level))
        ci_margin = k_multiplier * mae
        ci_lower = pred_series - ci_margin
        ci_upper = pred_series + ci_margin
        day_delta = 1.0
        velocity = full_series.diff().div(day_delta)
        acceleration = full_series.diff().div(day_delta)
        velocity = velocity.iloc[-len(pred_series) :]
        acceleration = acceleration.iloc[-len(pred_series) :]
        return ci_lower.values, ci_upper.values, velocity.values, acceleration.values

    def run_reconstruct_save_results_pipeline(
        self,
        input_directory: str,
        old_input_directory: str,
        code: str,
        LOOKBACK_LIST: List[int],
        FORECAST_LIST: List[int],
        final_output_predictions: Optional[np.ndarray],
        final_output_df: pd.DataFrame,
    ) -> pd.DataFrame:
        dates = ["2025-12-23", "2025-12-24", "2025-12-25", "2025-12-26", "2025-12-27", "2025-12-28", "2025-12-29", "2025-12-30", "2025-12-31"]
        for max_date in dates:
            for lookback, forecast in zip(LOOKBACK_LIST, FORECAST_LIST):
                auxiliary_output_df = pd.DataFrame()
                X_univ_data_production, Y_univ_data_production, df_timestamp_production = self.prepare_prediction_univ_data(
                    data_path=input_directory,
                    code=code,
                    lookback=lookback,
                    forecast=forecast,
                    cutoff_date=self.config.cutoff_date,
                    max_date=max_date,
                    scaler=self.config.scaler,
                    eliminate_covid_data=self.config.eliminate_covid_data,
                    covid_token=self.config.covid_token,
                    production_mode=True,
                )
                X_diagnostics_data_production, Y_diagnostics_data_production = self.prepare_prediction_diagnostics_data(
                    data_path=old_input_directory,
                    code=code,
                    lookback=lookback,
                    forecast=forecast,
                    cutoff_date=self.config.cutoff_date,
                    max_date=max_date,
                    scaler=self.config.scaler,
                    covid_token=self.config.covid_token,
                    production_mode=True,
                )
                X_seasonal_data_production = self.prepare_prediction_seasonal_data(
                    data_path=input_directory,
                    code=code,
                    forecast=forecast,
                    lookback=lookback,
                    cutoff_date=self.config.cutoff_date,
                    max_date=max_date,
                    categorical_vars=self.config.DEFAULT_SEASONAL_CATEGORICAL_VARS,
                    predictions_train=None,
                    predictions_test=None,
                    scaler=self.config.scaler,
                )
                X_univ_data, Y_univ_data, df_timestamp = self.prepare_prediction_univ_data(
                    data_path=input_directory,
                    code=code,
                    lookback=lookback,
                    forecast=forecast,
                    cutoff_date=self.config.cutoff_date,
                    max_date=max_date,
                    scaler=self.config.scaler,
                    eliminate_covid_data=self.config.eliminate_covid_data,
                    covid_token=self.config.covid_token,
                    production_mode=False,
                )
                X_diagnostics_data, Y_diagnostics_data = self.prepare_prediction_diagnostics_data(
                    data_path=old_input_directory,
                    code=code,
                    lookback=lookback,
                    forecast=forecast,
                    cutoff_date=self.config.cutoff_date,
                    max_date=max_date,
                    scaler=self.config.scaler,
                    covid_token=self.config.covid_token,
                    production_mode=False,
                )
                X_seasonal_data = self.prepare_prediction_seasonal_data(
                    data_path=input_directory,
                    code=code,
                    forecast=forecast,
                    lookback=lookback,
                    cutoff_date=self.config.cutoff_date,
                    max_date=max_date,
                    categorical_vars=self.config.DEFAULT_SEASONAL_CATEGORICAL_VARS,
                    predictions_train=None,
                    predictions_test=None,
                    scaler=self.config.scaler,
                )

                univ_input_shape = X_univ_data.shape[1:]
                diagnostics_input_shape = X_diagnostics_data.shape[1:]
                seasonal_input_shape = X_seasonal_data.shape
                seasonal_input_shape = tuple((seasonal_input_shape[1], seasonal_input_shape[2] + 1))

                univ_model, diagnostics_model, seasonal_model = self.reconstruct_full_model(
                    code,
                    lookback,
                    forecast,
                    models_directory=self.config.model_folder,
                    univ_input_shape=univ_input_shape,
                    diagnostics_input_shape=diagnostics_input_shape,
                    seasonal_input_shape=seasonal_input_shape,
                    head_size=self.config.head_size,
                    num_heads=self.config.num_heads,
                    ff_dim=self.config.ff_dim,
                    num_transformer_blocks=self.config.num_transformer_blocks,
                    mlp_units=self.config.mlp_units,
                    activation_function=self.config.activation_function,
                    dropout=self.config.dropout,
                    mlp_dropout=0,
                    n_pred=1,
                    pos_encoding=True,
                )

                quant_predictions_univ = univ_model.predict(X_univ_data, verbose=0)
                quant_predictions_diagnostics_residuals = diagnostics_model.predict(X_diagnostics_data, verbose=0)
                preds_reshaped = quant_predictions_diagnostics_residuals[:, -forecast:, np.newaxis]
                X_seasonal_with_preds = np.concatenate((X_seasonal_data, preds_reshaped), axis=2)
                quant_predictions_seasonal_residuals = seasonal_model.predict(X_seasonal_with_preds, verbose=0)

                quant_pred_corrected_diagnostics = quant_predictions_univ + quant_predictions_diagnostics_residuals
                quant_pred_corrected_seasonal = quant_pred_corrected_diagnostics + quant_predictions_seasonal_residuals
                quant_pred_corrected_seasonal = np.clip(quant_pred_corrected_seasonal, 0, np.inf)

                mae = np.mean(np.abs(quant_pred_corrected_seasonal - Y_univ_data))
                mse = np.mean((quant_pred_corrected_seasonal - Y_univ_data) ** 2)
                wape = np.sum(np.abs(quant_pred_corrected_seasonal - Y_univ_data)) / np.sum(np.abs(Y_univ_data) + 1e-8)
                print(f"MAE for code {code} with lookback {lookback} and forecast {forecast}: {mae}")
                print(f"MSE for code {code} with lookback {lookback} and forecast {forecast}: {mse}")
                print(f"WAPE for code {code} with lookback {lookback} and forecast {forecast}: {wape * 100:.2f}%")

                quant_predictions_univ_production = univ_model.predict(X_univ_data_production, verbose=0)
                quant_predictions_diagnostics_residuals_production = diagnostics_model.predict(X_diagnostics_data_production, verbose=0)
                preds_reshaped_production = quant_predictions_diagnostics_residuals_production[:, -forecast:, np.newaxis]
                X_seasonal_with_preds = np.concatenate((X_seasonal_data_production, preds_reshaped_production), axis=2)
                quant_predictions_seasonal_residuals_production = seasonal_model.predict(X_seasonal_with_preds, verbose=0)

                quant_pred_corrected_diagnostics_production = quant_predictions_univ_production + quant_predictions_diagnostics_residuals_production
                quant_pred_corrected_seasonal_production = quant_pred_corrected_diagnostics_production + quant_predictions_seasonal_residuals_production
                quant_pred_corrected_seasonal_production = np.clip(quant_pred_corrected_seasonal_production, 0, np.inf)

                ci_lower, ci_upper, velocity, acceleration = self.compute_predap_auxiliary_metrics(
                    quant_pred_corrected_seasonal_production.flatten(),
                    Y_univ_data.flatten(),
                    mae,
                    lookback=lookback,
                    confidence_level=0.95,
                )

                target_date = df_timestamp_production[-forecast:].values
                init_forecast_date = df_timestamp.iloc[-forecast - 1]
                final_forecast_date = df_timestamp.iloc[-1]

                auxiliary_output_df["code"] = [code] * forecast
                auxiliary_output_df["target_date"] = target_date
                auxiliary_output_df["init_forecast_date"] = [init_forecast_date] * forecast
                auxiliary_output_df["final_forecast_date"] = [final_forecast_date] * forecast
                auxiliary_output_df["forecast"] = [forecast] * forecast
                auxiliary_output_df["lookback"] = [lookback] * forecast
                auxiliary_output_df["predictions"] = quant_pred_corrected_seasonal_production.flatten()
                auxiliary_output_df["ci_lower"] = ci_lower
                auxiliary_output_df["ci_upper"] = ci_upper
                auxiliary_output_df["velocity"] = velocity
                auxiliary_output_df["acceleration"] = acceleration

                final_output_df = pd.concat([final_output_df, auxiliary_output_df], ignore_index=True)

        return final_output_df

    def save_final_output_predictions(self, final_output_df: pd.DataFrame, output_path: str = "../production_predictions/final_output_predictions"):
        table = pa.Table.from_pandas(final_output_df, preserve_index=False)
        output_path = self.config.production_predictions_dir if output_path is None else output_path
        ds.write_dataset(
            table,
            base_dir=output_path,
            format="parquet",
            partitioning=["code"],
            existing_data_behavior="overwrite_or_ignore",
        )
        return output_path

    def delete_old_data(self, predictions_dataset_path: str, real_data_dataset_path: Optional[str] = None, metrics_df_path: str = "../production_predictions/production_evaluation_metrics.parquet"):
        if not predictions_dataset_path:
            return None
        if not tf.io.gfile.exists(predictions_dataset_path):
            print(f"No data file found at: {predictions_dataset_path}")
            return None
        print("Inference bundle keeps the saved prediction dataset as-is.")
        return predictions_dataset_path


def get_codes_list(input_directory: str) -> List[str]:
    if input_directory.endswith(".csv"):
        df = pd.read_csv(input_directory, nrows=1000)
    elif input_directory.endswith(".parquet"):
        df = pd.read_parquet(input_directory, engine="pyarrow")
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Parquet file.")
    codes_list = df.columns.tolist()
    codes_list.remove("timestamp")
    return codes_list
