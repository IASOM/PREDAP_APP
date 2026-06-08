import shutil

from sklearn.preprocessing import FunctionTransformer
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import sys
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Optional, Dict, List, Tuple, Any
import pyarrow as pa
import pyarrow.dataset as ds

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.experiments_utils import smart_read
from model_architechture import model_architecture_univ_transformer
from model_architechture import model_architecture_residual_transformer
from config.base_transformer_config import BaseTransformerConfig
from production.data_preparation_in_poduction import DataPreparationInProduction

np.read_csv = smart_read

default_config = BaseTransformerConfig()


class ModelPredictionPipeline(DataPreparationInProduction):
    def __init__(self, config: BaseTransformerConfig):
        self.config = config
        self.config.print_config()

    def create_univariate_transformer_model(self, input_shape, forecast, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, activation_function="tanh", dropout=0, mlp_dropout=0, n_pred=1, pos_encoding=True):
        """Creates a univariate transformer model for time series forecasting.
        Args:
            input_shape (tuple): The shape of the input data (lookback, num_features).
            forecast (int): The number of future time steps to predict.
            head_size (int): The dimensionality of the attention heads.
            num_heads (int): The number of attention heads.
            ff_dim (int): The dimensionality of the feed-forward layer.
            num_transformer_blocks (int): The number of transformer blocks to stack.
            mlp_units (list of int): A list specifying the number of units in each MLP layer after the transformer blocks.
            activation_function (str): The activation function to use in the transformer and MLP layers.
            dropout (float): The dropout rate for regularization in the transformer blocks.
            mlp_dropout (float): The dropout rate for regularization in the MLP layers.
            n_pred (int): The number of future time steps to predict (forecast horizon).
            pos_encoding (bool): Whether to use positional encoding in the model.
        Returns:
            A compiled Keras model instance representing the univariate transformer.
        """
        model = model_architecture_univ_transformer.build_model(
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
            pos_encoding=pos_encoding
        )
        return model

    def create_residual_transformer_model(self, input_shape, forecast, transformer_params, activation_function="tanh"):
        """Creates a residual transformer model for either diagnostics or seasonal components.
        Args:
            input_shape (tuple): The shape of the input data (lookback, num_features).
            forecast (int): The number of future time steps to predict.
            transformer_params (dict): A dictionary containing the parameters for the transformer architecture, including:
                - head_size (int): The dimensionality of the attention heads.
                - num_heads (int): The number of attention heads.
                - ff_dim (int): The dimensionality of the feed-forward layer.
                - mlp_units (list of int): A list specifying the number of units in each MLP layer after the transformer blocks.
                - num_transformer_blocks (int): The number of transformer blocks to stack.
            activation_function (str): The activation function to use in the transformer and MLP layers.
        Returns:
            A compiled Keras model instance representing the residual transformer."""
        model = model_architecture_residual_transformer.hybrid_lstm_transformer_model(
            input_shape=input_shape,
            forecast = forecast,
            transformer_params=transformer_params,
            activation_function=activation_function
        )
        return model
    
    def load_model_weights(self,model, code, lookback, forecast, models_directory, model_type="univariate"):
        """
        Loads the weights of a Keras model from a specified path.
        
        Args:
            model: A compiled Keras model instance.
            model_path: The file path to the saved Keras model weights (e.g., .keras or .h5 file).
        Returns:
            The Keras model instance with loaded weights.
        """

        weights_path =  f"{models_directory}/{code}/{model_type}/{code}_{model_type}_{forecast}fh_{lookback}lb_f16_weights.h5"
        
        print(model.summary())
        # Try to identify mismatch before loading
        
        try:
            model.load_weights(weights_path)
            print(f"Successfully loaded weights from {weights_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")
            raise e  # Re-raise after logging
        return model

    def reconstruct_full_model(self, code: str, lookback: int, forecast: int, models_directory: str, univ_input_shape: Tuple[int], diagnostics_input_shape: Tuple[int], seasonal_input_shape: Tuple[int], head_size: int, num_heads: int, ff_dim: int, num_transformer_blocks: int, mlp_units: List[int], activation_function: str="tanh", dropout: float=0, mlp_dropout: float=0, n_pred: int=1, pos_encoding=True):
        """Reconstructs the full model architecture for univariate, diagnostics residual, and seasonal residual components, and loads the corresponding weights for each sub-model.
        
        Returns:
            A tuple containing the reconstructed univariate model, diagnostics residual model, and seasonal residual model with loaded weights.
        """
        
        DEFAULT_RESIDUAL_TRANSFORMER_PARAMS =  {
            'head_size': 16,
            'num_heads': 16,
            'ff_dim': 512,
            'mlp_units': [256, 128],
            'num_transformer_blocks': 2,
            'dropout': 0
        }
        
        
        univ_model = self.create_univariate_transformer_model(
            input_shape= univ_input_shape,
            forecast= forecast,
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
            input_shape= diagnostics_input_shape,
            forecast = forecast,
            transformer_params=DEFAULT_RESIDUAL_TRANSFORMER_PARAMS, 
            activation_function=activation_function
        )



        seasonal_model = self.create_residual_transformer_model(
            input_shape= seasonal_input_shape,
            forecast = forecast,
            transformer_params=DEFAULT_RESIDUAL_TRANSFORMER_PARAMS,
            activation_function=activation_function
        )


        #Temporary solution for the name mismatch between the codes in the input data and the codes expected by the reconstruction pipeline and the saved models. Remove the first "DEMAND_" characters if they are present, and replace any "_" with "__" to match the format of the saved models. 
        def process_col_name(col):
            # 1. Keep 'timestamp' as is
            if col == 'timestamp':
                return col
            
            # 2. Remove 'DEMAND_' prefix if present
            code = col[7:] if col.startswith("DEMAND_") else col
            
            # 3. Identify all positions of '_'
            indices = [i for i, char in enumerate(code) if char == '_']
            
            # If no underscores, return code as is
            if not indices:
                return code
            
            # If there is only one underscore, replace it with '__'
            if len(indices) == 1:
                idx = indices[0]
                return code[:idx] + "__" + code[idx+1:]
            
            # If there are multiple, replace the first and last with '__'
            first_idx = indices[0]
            last_idx = indices[-1]
            
            # Reconstruct the string: 
            # Prefix + "__" + Middle + "__" + Suffix
            return code[:first_idx] + "__" + code[first_idx+1:last_idx] + "__" + code[last_idx+1:]

        #Temporary solution for the name mismatch 
        code = code[7:] if code.startswith("DEMAND_") else code
        #add a __ if you find a _ in the code, to match the format of the saved models, which have _ instead of : in the code names
        code = process_col_name(code)


        univ_model = self.load_model_weights(univ_model, code, lookback, forecast, models_directory, model_type="univariate_model")
        diagnostics_model = self.load_model_weights(diagnostics_model, code, lookback, forecast, models_directory, model_type="diagnostics_model")
        seasonal_model = self.load_model_weights(seasonal_model, code, lookback, forecast, models_directory, model_type="seasonal_model")

        return univ_model, diagnostics_model, seasonal_model

    def compute_predap_auxiliary_metrics(self, predictions: np.ndarray, true_past_data: np.ndarray, mae: np.ndarray, lookback: int, confidence_level: float=0.95) -> Dict[str, float]:
        """
        Computes auxiliary evaluation metrics (MAE, MSE, WAPE) for the given predictions and true values.

        Args:
            predictions (np.ndarray): The array of predicted values.
            true_past_data (np.ndarray): The array of true past values.
            mae (np.ndarray): The mean absolute error.
            lookback (int): The lookback period.
            confidence_level (float): The confidence level for the confidence interval.
        Returns:
            
        """

        pred_series = pd.Series(predictions.flatten())
        true_past_data = pd.Series(true_past_data.flatten())
        full_series = pd.concat([true_past_data, pred_series], ignore_index=True)

        # 4. Calculate the Laplace multiplier (k) instead of Z-score
        k_multiplier = -np.log(2 * (1.0 - confidence_level))

        # 5. Compute the final bounds centered around your current predictions
        ci_margin = k_multiplier * mae

        ci_lower = pred_series - ci_margin
        ci_upper = pred_series + ci_margin

        #compute 1st and 2nd derivatives (velocity and acceleration)
        day_delta = 1.0
        velocity = full_series.diff().div(day_delta)
        acceleration = full_series.diff().div(day_delta)

        velocity = velocity.iloc[-len(pred_series):]  # Keep only the velocity for the predicted points
        acceleration = acceleration.iloc[-len(pred_series):]  # Keep only the acceleration for the predicted points

        return ci_lower.values, ci_upper.values, velocity.values, acceleration.values

    
    
    
    
    def run_reconstruct_save_results_pipeline(
            self, 
            input_directory: str,
            old_input_directory: str, #Temporary solution for the name mismatch between the codes in the input data and the codes expected by the reconstruction pipeline and the saved models. Remove the first "DEMAND_" characters if they are present, and replace any "_" with "__" to match the format of the saved models.
            code: str, 
            LOOKBACK_LIST: List[int], 
            FORECAST_LIST: List[int], 
            final_output_predictions: Optional[np.ndarray], 
            final_output_df: pd.DataFrame,
            prediction_dates: Optional[List[str]] = None,
            
            ) -> pd.DataFrame:
        """Runs the full pipeline to reconstruct the model, make predictions, and save results for a given code and list of lookback and forecast combinations. 
        Args:
            code (str): The code for which to run the pipeline (e.g., 'demanda__TOTAL').
            LOOKBACK_LIST (List[int]): A list of lookback periods to iterate
            FORECAST_LIST (List[int]): A list of forecast horizons to iterate.
            final_output_predictions (Optional[np.ndarray]): An optional array to store final output predictions across iterations
            final_output_df (pd.DataFrame): A DataFrame to store the final output predictions along with corresponding dates.
        Returns:
            A DataFrame containing the final output predictions for each forecast horizon along with corresponding dates.
        """
        if final_output_df is None:
            final_output_df = pd.DataFrame()
        if len(LOOKBACK_LIST) != len(FORECAST_LIST):
            raise ValueError("LOOKBACK_LIST and FORECAST_LIST must have the same length")
        
        dates = prediction_dates or ["2025-12-23","2025-12-24", "2025-12-25", "2025-12-26", "2025-12-27", "2025-12-28", "2025-12-29", "2025-12-30", "2025-12-31"]
        for max_date in dates:
            for lookback, forecast in zip(LOOKBACK_LIST, FORECAST_LIST):
                auxiliary_output_df = pd.DataFrame()  # Temporary DataFrame for current iteration
                
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
                    production_mode=True
                )

                X_diagnostics_data_production, Y_diagnostics_data_production = self.prepare_prediction_diagnostics_data(
                    data_path=old_input_directory, #Temporary solution for the name mismatch between the codes in the input data and the codes expected by the reconstruction pipeline and the saved models. Use the old input directory which has the original code names with "DEMAND_" prefix and "_" instead of "__".
                    code=code,
                    lookback=lookback,
                    forecast=forecast,
                    cutoff_date = self.config.cutoff_date,
                    max_date=max_date,
                    scaler=self.config.scaler,
                    covid_token=self.config.covid_token,
                    production_mode=True
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
                    production_mode=False
                )

                X_diagnostics_data, Y_diagnostics_data = self.prepare_prediction_diagnostics_data(
                    data_path=old_input_directory, #Temporary solution for the name mismatch between the codes in the input data and the codes expected by the reconstruction pipeline and the saved models. Use the old input directory which has the original code names with "DEMAND_" prefix and "_" instead of "__".
                    code=code,
                    lookback=lookback,
                    forecast=forecast,
                    cutoff_date=self.config.cutoff_date,
                    max_date=max_date,
                    scaler=self.config.scaler,
                    covid_token=self.config.covid_token,
                    production_mode=False
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
                seasonal_input_shape = tuple((seasonal_input_shape[1], seasonal_input_shape[2] + 1))  # Add 1 to the last dimension to account for the diagnostics predictions that will be concatenated as an additional feature

                
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
                    pos_encoding=True
                )

                #Make predictions for the known historical data
                quant_predictions_univ = univ_model.predict(X_univ_data, verbose=0)
                quant_predictions_diagnostics_residuals = diagnostics_model.predict(X_diagnostics_data, verbose=0)
                
                preds_reshaped = quant_predictions_diagnostics_residuals[:, -forecast:, np.newaxis]
                X_seasonal_with_preds = np.concatenate((X_seasonal_data, preds_reshaped), axis=2)
                #X_seasonal_with_preds = X_seasonal_data.copy() #temporary put only the original seasonal data, without the diagnostics predictions, to avoid potential issues with input shape or model expectations.
                quant_predictions_seasonal_residuals = seasonal_model.predict(X_seasonal_with_preds, verbose=0)


                quant_pred_corrected_diagnostics = quant_predictions_univ + quant_predictions_diagnostics_residuals
                quant_pred_corrected_seasonal = quant_pred_corrected_diagnostics + quant_predictions_seasonal_residuals
                #Clip predictions to be non-negative, as we are predicting production values that cannot be negative
                quant_pred_corrected_seasonal = np.clip(quant_pred_corrected_seasonal, 0, +np.inf)

                mae = np.mean(np.abs(quant_pred_corrected_seasonal - Y_univ_data))
                mse = np.mean((quant_pred_corrected_seasonal - Y_univ_data)**2)
                wape = np.sum(np.abs(quant_pred_corrected_seasonal - Y_univ_data)) / np.sum(np.abs(Y_univ_data) + 1e-8)
                print(f"MAE for code {code} with lookback {lookback} and forecast {forecast}: {mae}")
                print(f"MSE for code {code} with lookback {lookback} and forecast {forecast}: {mse}")
                print(f"WAPE for code {code} with lookback {lookback} and forecast {forecast}: {wape*100:.2f}%")

                #Make predictions for the unknown historical data
                quant_predictions_univ_production = univ_model.predict(X_univ_data_production, verbose=0)
                quant_predictions_diagnostics_residuals_production = diagnostics_model.predict(X_diagnostics_data_production, verbose=0)
                
                preds_reshaped_production = quant_predictions_diagnostics_residuals_production[:, -forecast:, np.newaxis]
                X_seasonal_with_preds = np.concatenate((X_seasonal_data_production, preds_reshaped_production), axis=2)
                #X_seasonal_with_preds = X_seasonal_data.copy() #temporary put only the original seasonal data, without the diagnostics predictions, to avoid potential issues with input shape or model expectations.
                quant_predictions_seasonal_residuals_production = seasonal_model.predict(X_seasonal_with_preds, verbose=0)


                quant_pred_corrected_diagnostics_production = quant_predictions_univ_production + quant_predictions_diagnostics_residuals_production
                quant_pred_corrected_seasonal_production = quant_pred_corrected_diagnostics_production + quant_predictions_seasonal_residuals_production
                #Clip predictions to be non-negative, as we are predicting production values that cannot be negative
                quant_pred_corrected_seasonal_production = np.clip(quant_pred_corrected_seasonal_production, 0, +np.inf)
                

                ci_lower, ci_upper, velocity, acceleration = self.compute_predap_auxiliary_metrics(quant_pred_corrected_seasonal_production.flatten(), Y_univ_data.flatten(), mae, lookback=lookback, confidence_level=0.95)

                target_date = df_timestamp_production[-forecast:].values
                init_forecast_date = df_timestamp.iloc[-forecast-1]
                final_forecast_date = df_timestamp.iloc[-1]

                auxiliary_output_df["target_date"] = target_date
                auxiliary_output_df["init_forecast_date"] = init_forecast_date
                auxiliary_output_df["final_forecast_date"] = final_forecast_date
                auxiliary_output_df["code"] = code
                auxiliary_output_df["forecast"] = forecast
                auxiliary_output_df["predictions"] = quant_pred_corrected_seasonal_production.flatten()[:forecast]
                auxiliary_output_df["ci_lower"] = ci_lower
                auxiliary_output_df["ci_upper"] = ci_upper
                auxiliary_output_df["velocity"] = velocity
                auxiliary_output_df["acceleration"] = acceleration

                final_output_df = pd.concat([final_output_df, auxiliary_output_df], ignore_index=True)
            
        #final_output_df.to_csv(f"../quantized_models/{code}/final_output_predictions_{code.replace(':', '#')}.csv", index=False)

        return final_output_df

    def save_final_output_predictions(self, final_output_df: pd.DataFrame, output_path: Optional[str] = None):
        """
        Saves the final output predictions DataFrame to a parquet file.

        Args:
            final_output_df (pd.DataFrame): The DataFrame containing the final output predictions along with corresponding dates.
            code (str): The code for which the predictions were made, used for naming the output file.
        """
        
        if final_output_df is None:
            raise ValueError("final_output_df cannot be None")
        output_path = output_path or self.config.production_predictions_dir
        table = pa.Table.from_pandas(final_output_df, preserve_index=False)

        ds.write_dataset(
            table,
            base_dir=output_path,
            format="parquet",
            partitioning=["code"],
            existing_data_behavior="overwrite_or_ignore"
        )

    def compute_code_prediction_features(
        self,
        final_output_df: pd.DataFrame,
        code: str,
        lookback: int,
        real_data_df: Optional[pd.DataFrame] = None,
        confidence_level: float = 0.95,
    ) -> pd.DataFrame:
        """
        Computes prediction features for a single code from final_output_df:
        1) Confidence interval using the last rolling window of size lookback.
        2) WAPE for predictions (when real values are available).
        3) First and second derivatives (velocity and acceleration) of predictions.

        Args:
            final_output_df (pd.DataFrame): DataFrame with at least ['code', 'target_date', 'predictions'].
            code (str): Single code to process.
            lookback (int): Rolling window size used for confidence interval estimation.
            real_data_df (Optional[pd.DataFrame]): Optional DataFrame with real values.
                Expected columns: ['timestamp', <real_code_column>], where <real_code_column>
                is code with '#' replaced by ':'.
            confidence_level (float): Confidence level for CI (default 0.95).

        Returns:
            pd.DataFrame: Code-filtered DataFrame enriched with CI, WAPE, velocity,
                and acceleration columns.
        """
        required_cols = {"code", "target_date", "predictions"}
        missing_cols = required_cols - set(final_output_df.columns)
        if missing_cols:
            raise ValueError(
                f"Missing required columns in final_output_df: {sorted(missing_cols)}"
            )
        if lookback <= 0:
            raise ValueError("lookback must be a positive integer")

        df_code = final_output_df.loc[final_output_df["code"] == code].copy()
        if df_code.empty:
            raise ValueError(f"No rows found for code '{code}' in final_output_df")

        df_code["target_date"] = pd.to_datetime(df_code["target_date"], errors="coerce")
        if "forecast_date" in df_code.columns:
            df_code["forecast_date"] = pd.to_datetime(df_code["forecast_date"], errors="coerce")
            df_code = df_code.sort_values(["forecast_date", "target_date"]).reset_index(drop=True)
        else:
            df_code = df_code.sort_values("target_date").reset_index(drop=True)

        # CI is computed from the last lookback predictions (excluding current point).
        pred_series = pd.to_numeric(df_code["predictions"], errors="coerce")
        rolling_mean = pred_series.shift(1).rolling(window=lookback, min_periods=1).mean()
        rolling_std = pred_series.shift(1).rolling(window=lookback, min_periods=2).std(ddof=1).fillna(0.0)

        alpha_tail = (1.0 + confidence_level) / 2.0
        z_score = NormalDist().inv_cdf(alpha_tail)

        ci_margin = z_score * rolling_std
        df_code["ci_window_mean"] = rolling_mean
        df_code["ci_window_std"] = rolling_std
        df_code["ci_lower"] = pred_series - ci_margin
        df_code["ci_upper"] = pred_series + ci_margin

        # First and second derivatives over time (daily units).
        day_delta = (
            df_code["target_date"].diff().dt.total_seconds().div(86400.0).replace(0, np.nan)
        )
        df_code["velocity"] = pred_series.diff().div(day_delta)
        df_code["acceleration"] = df_code["velocity"].diff().div(day_delta)

        # WAPE computation when actuals are available.
        if "actual" in df_code.columns:
            df_code["actual"] = pd.to_numeric(df_code["actual"], errors="coerce")
        else:
            df_code["actual"] = np.nan

        if real_data_df is not None:
            real_code = code.replace("#", ":")
            if "timestamp" not in real_data_df.columns:
                raise ValueError("real_data_df must contain a 'timestamp' column")
            if real_code not in real_data_df.columns:
                raise ValueError(
                    f"real_data_df must contain column '{real_code}' for code '{code}'"
                )

            real_slice = real_data_df[["timestamp", real_code]].copy()
            real_slice["timestamp"] = pd.to_datetime(real_slice["timestamp"], errors="coerce")
            real_slice = real_slice.rename(columns={"timestamp": "target_date", real_code: "actual"})

            df_code = df_code.merge(real_slice, on="target_date", how="left", suffixes=("", "_real"))
            if "actual_real" in df_code.columns:
                df_code["actual"] = df_code["actual_real"]
                df_code = df_code.drop(columns=["actual_real"])

        df_code["wape_row"] = np.nan
        df_code["wape_code"] = np.nan
        valid_actuals = df_code["actual"].notna()
        if valid_actuals.any():
            abs_error = np.abs(
                pd.to_numeric(df_code.loc[valid_actuals, "predictions"], errors="coerce")
                - df_code.loc[valid_actuals, "actual"]
            )
            abs_actual = np.abs(df_code.loc[valid_actuals, "actual"]) + 1e-8
            df_code.loc[valid_actuals, "wape_row"] = abs_error / abs_actual

            denom = np.abs(df_code.loc[valid_actuals, "actual"]).sum() + 1e-8
            df_code["wape_code"] = abs_error.sum() / denom

        return df_code
    

    def compute_evaluation_metrics(self, predictions_df: np.ndarray, real_data_df: np.ndarray, metrics_df: pd.DataFrame) -> Dict[str, float]:
        """
        Computes evaluation metrics (MAE, MSE, WAPE) between predictions and real data.

        Args:
            predictions_df (np.ndarray): The array of predicted values.
            real_data_df (np.ndarray): The array of actual values.
        Returns:
            A dictionary containing the computed evaluation metrics.
        """

        codes = np.unique(predictions_df['code'].values)
        dates = np.unique(predictions_df['target_date'].values)
        forecasts = np.unique(predictions_df['forecast'].values[0])
        results_accumulator = []
        for code in codes:
            for date in dates:
                for forecast in forecasts:
                    real_code = code.replace('#', ':')
                    preds_mask = (predictions_df['code'] == code) & (predictions_df['target_date'] == date) & (predictions_df['forecast'] == forecast)
                    preds = predictions_df.loc[preds_mask, 'predictions'].values
                    
                    real_data_df['timestamp'] = pd.to_datetime(real_data_df['timestamp'])
                    real_mask = real_data_df[real_code][real_data_df['timestamp'] == date].index
                    real = real_data_df.loc[real_mask, real_code].values

                    mae = np.mean(np.abs(preds - real))
                    mse = np.mean((preds - real)**2)
                    rmse = np.sqrt(mse)
                    real_lb_data = real_data_df[(real_data_df['timestamp'] < date) & (real_data_df['timestamp'] >= (pd.to_datetime(date) - pd.Timedelta(days= forecast)))]
                    wape = np.sum(np.abs(preds - real)) / np.sum(np.abs(real_lb_data[real_code]) + 1e-8)

                    print(f"Evaluation for code {code} on date {date}: MAE={mae}, MSE={mse}, WAPE={wape*100:.2f}%")
                    results_accumulator.append({
                        'code': code,
                        'date': date,
                        'forecast': forecast,
                        'MAE': mae,
                        'MSE': mse,
                        'RMSE': rmse,
                        'WAPE': wape *100   

                    })
        metrics_df = pd.concat([metrics_df, pd.DataFrame(results_accumulator)], ignore_index=True)
        output_path = self.config.production_metrics_file
        metrics_df.to_parquet(output_path, index=False)
        return metrics_df

        

    def delete_old_data(self, predictions_dataset_path: str, real_data_dataset_path: Optional[str] = None, metrics_df_path: str = "../production_predictions/production_evaluation_metrics.parquet"):
        """
        Deletes rows from a parquet file where the day difference between
        target_date and forecast_date equals the value in forecast.

        Args:
            predictions_dataset_path (str): Full parquet dataset path created by save_final_output_predictions.
            real_data_dataset_path (Optional[str]): Optional path to the real data parquet file for additional processing.
            metrics_df_path (str): Path to the existing metrics dataframe. If it doesn't exist, a new one will be created.
        
        """
        if not os.path.exists(predictions_dataset_path):
            print(f"No data file found at: {predictions_dataset_path}")
            return None
        if not os.path.exists(real_data_dataset_path):
            print(f"No real data file found at: {real_data_dataset_path}")
        else:
            print(f"Real data file found at: {real_data_dataset_path}. It will be used for additional processing.")

        if os.path.exists(metrics_df_path):
            metrics_df = pd.read_parquet(metrics_df_path)
            print(f"Existing metrics dataframe loaded from: {metrics_df_path}")
        else:
            print(f"No existing metrics dataframe found at: {metrics_df_path}. A new one will be created.")
            metrics_df = pd.DataFrame(columns=['code', 'date', 'MAE', 'MSE', 'RMSE', 'WAPE'])
            
        real_data_dataset = smart_read(real_data_dataset_path) if real_data_dataset_path else None

        my_schema = pa.schema([("code", pa.string())])

        # 2. Create the partitioning object WITH the flavor
        partition_schema = ds.partitioning(my_schema)
        dataset = ds.dataset(predictions_dataset_path, format="parquet", partitioning=partition_schema)

        
        table = dataset.to_table()
        df = table.to_pandas()

        required_cols = {"target_date", "final_forecast_date", "forecast"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in {predictions_dataset_path}: {sorted(missing)}"
            )

        # Ensure datetime + numeric types
        df["target_date"] = pd.to_datetime(df["target_date"], errors="coerce")
        df["final_forecast_date"] = pd.to_datetime(df["final_forecast_date"], errors="coerce")
        forecast_days = pd.to_numeric(df["forecast"], errors="coerce")  # Add 1 to match the day difference logic (inclusive of start date)

        # Rows to delete:
        # difference in days equals forecast value in that row
        #diff_days = (df["target_date"] - df["final_forecast_date"]).dt.days
        # 1. Gather ALL unique forecast dates from the entire dataset
        all_forecast_dates = pd.concat(
            [df["init_forecast_date"], df["final_forecast_date"]]
        ).dropna().unique()

        # 2. Condition A: Does the target_date match ANY forecast date in the dataset?
        matches_forecast = df["target_date"].isin(all_forecast_dates)

        # 4. Combine both conditions using a bitwise AND (&)
        delete_mask = matches_forecast 

        deleted_rows = int(delete_mask.sum())
        df_to_delete = df.loc[delete_mask].copy()

        if real_data_dataset is not None and not df_to_delete.empty:
            self.compute_evaluation_metrics(predictions_df=df_to_delete, real_data_df=real_data_dataset, metrics_df=metrics_df)
        else:
            print("No rows to delete or real data dataset not available. Skipping evaluation metrics computation.")

        df_clean = df.loc[~delete_mask].copy()

        if os.path.isdir(predictions_dataset_path):
            # 2. Delete the directory to clear old partitions
            shutil.rmtree(predictions_dataset_path)

        table = pa.Table.from_pandas(df_clean)
        write_partitioning = ds.partitioning(pa.schema([("code", pa.string())]))
        ds.write_dataset(
            table, 
            base_dir=predictions_dataset_path, 
            format="parquet", 
            partitioning=write_partitioning,
            existing_data_behavior="overwrite_or_ignore"
        )

        '''df_clean.to_parquet(
            predictions_dataset_path, 
            engine='pyarrow', 
            partition_cols=['code'], 
            index=False
        )'''
        print(f"Deleted {deleted_rows} rows from: {predictions_dataset_path}")
        print(f"Remaining rows: {len(df_clean)}")

        return predictions_dataset_path
    

if __name__ == "__main__":
    CODES_LIST = ["demanda__SERVEI_CODI__INF",
                    "demanda__SERVEI_CODI__INFP",
                    "demanda__SERVEI_CODI__MF",
                    "demanda__SERVEI_CODI__PED",
                    "demanda__SERVEI_CODI__URG",
                    "demanda__TIPUS_CLASS__9T",
                    "demanda__TIPUS_CLASS__C9C",
                    "demanda__TIPUS_CLASS__C9R",
                    ]
    
                    #['demanda__TOTAL', 'demanda__SERVEI_CODI__URG', 'B34','J00', 'I10', 'M54','Ch01#subch01#A00-A09']
    LOOKBACK_LIST = [7,14, 60, 60, 182,182]
    FORECAST_LIST = [7,14, 30, 60, 182,365]
    FINAL_LOOKBACK = 182
    FINAL_FORECAST = 365

    #input_directory = '../data/FINAL_DB/full_CAT1.parquet'
    input_directory = '../data/FINAL_DB/finals_combined.csv'
    #model_folder = '../transformer_outputs/models_covid_token'
    output_path = f"../production_predictions/final_output_predictions"
    metrics_df_path = "../production_predictions/production_evaluation_metrics.parquet"
    scaler = FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)
    max_date = '2027-09-30'
    cutoff_date = '2008-01-01'
    eliminate_covid_data = False
    covid_dates = None
    model_folder = "../quantized_models"
    head_size = 32
    num_heads = 8
    ff_dim = 512
    num_transformer_blocks = 2
    mlp_units = [512,256]
    activation_function = "gelu"


    final_output_df = pd.DataFrame()
    for code in CODES_LIST:
        final_output_predictions = None
        
        #final_output_df = generate_future_dates_df(input_directory, num_days=FORECAST_LIST[-1])
        
        base_pipeline = ModelPredictionPipeline(config=BaseTransformerConfig(
                code=code,
                head_size=head_size,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_transformer_blocks=num_transformer_blocks,
                mlp_units=mlp_units,
                activation_function=activation_function,
                dropout=0,
                learning_rate=0.001,
                epochs=50,
                batch_size=32,
                cutoff_date=cutoff_date,
                covid_token=True,
                positional_encoding=True,
                evaluate_model=True, 
                data_path=input_directory, 
                model_folder=model_folder,
            ))
        final_output_df = base_pipeline.run_reconstruct_save_results_pipeline(input_directory,input_directory, code, LOOKBACK_LIST, FORECAST_LIST, final_output_predictions, final_output_df)
        base_pipeline.save_final_output_predictions(final_output_df)
    base_pipeline.delete_old_data(predictions_dataset_path=output_path, real_data_dataset_path=input_directory, metrics_df_path=metrics_df_path)
    print(f"\nFinal output predictions for code {code}:\n")
    #print(final_output_df)
