
import numpy as np
import pandas as pd
import os
import re
from sklearn.preprocessing import FunctionTransformer
from typing import List, Optional, Tuple
from utils.experiments_utils import smart_read
from data_utils import data_preparation
from config.base_transformer_config import BaseTransformerConfig

default_config = BaseTransformerConfig()



class DataPreparationInProduction:
    def __init__(self, config: BaseTransformerConfig = default_config):
        self.config = config
        self.config.print_config()

    _COLUMN_ALIASES = {
        "VISI_SITUACIO_VISITA_N": "VISI_SITUACIO_VISITA_NO_PROGRAMADA",
        "VISI_SITUACIO_VISITA_P": "VISI_SITUACIO_VISITA_PROGRAMADA",
        "VISI_SITUACIO_VISITA_R": "VISI_SITUACIO_VISITA_URGENT",
        "SERVEI_CODI_MF": "SERVEI_CODI_MEDFAM",
    }

    def _canonical_column_name(self, name: str) -> str:
        canonical = str(name).strip().replace("#", ":")
        if canonical.startswith("DEMAND_"):
            canonical = canonical[len("DEMAND_"):]
        canonical = canonical.replace("__", "_")
        canonical = canonical.upper()
        for legacy, current in self._COLUMN_ALIASES.items():
            canonical = re.sub(
                rf"(^|_){re.escape(legacy)}($|_)",
                lambda match: f"{match.group(1)}{current}{match.group(2)}",
                canonical,
            )
        canonical = re.sub(r"[^A-Z0-9]+", "_", canonical).strip("_")
        return canonical

    def _column_lookup(self, columns) -> dict:
        lookup = {}
        for column in columns:
            lookup.setdefault(self._canonical_column_name(column), column)
        return lookup

    def _resolve_column(self, columns, requested: str, role: str) -> str:
        if requested in columns:
            return requested
        requested = str(requested).strip()
        lookup = self._column_lookup(columns)
        resolved = lookup.get(self._canonical_column_name(requested))
        if resolved is not None:
            if resolved != requested:
                print(f"-> INFO: Mapped {role} '{requested}' to dataset column '{resolved}'.")
            return resolved
        sample = ", ".join(map(str, list(columns)[:12]))
        raise ValueError(
            f"Could not map {role} '{requested}' to a column in the input dataset. "
            f"Available columns sample: {sample}"
        )

    def _resolve_columns(self, columns, requested_columns: List[str], role: str) -> List[str]:
        resolved_columns = []
        missing = []
        lookup = self._column_lookup(columns)
        for requested in requested_columns:
            requested = str(requested).strip()
            if not requested:
                continue
            if requested in columns:
                resolved_columns.append(requested)
                continue
            resolved = lookup.get(self._canonical_column_name(requested))
            if resolved is None:
                missing.append(requested)
            else:
                resolved_columns.append(resolved)
        if missing:
            examples = ", ".join(missing[:10])
            print(
                f"-> WARNING: Skipping {len(missing)} {role} not found in the input dataset. "
                f"First skipped values: {examples}."
            )
        changed = sum(
            1
            for original, resolved in zip(
                [str(col).strip() for col in requested_columns if str(col).strip()],
                resolved_columns,
            )
            if original != resolved
        )
        if changed:
            print(f"-> INFO: Mapped {changed} {role} to current dataset column names.")
        return resolved_columns

    def _resolve_feature_values(
        self,
        df_features: pd.DataFrame,
        requested_columns: List[str],
        role: str,
    ) -> np.ndarray:
        values = []
        missing = []
        changed = 0
        lookup = self._column_lookup(df_features.columns)
        for requested in requested_columns:
            requested = str(requested).strip()
            if not requested:
                continue
            if requested in df_features.columns:
                values.append(df_features[requested].values)
                continue
            resolved = lookup.get(self._canonical_column_name(requested))
            if resolved is None:
                missing.append(requested)
                values.append(np.zeros(len(df_features), dtype=np.float32))
            else:
                values.append(df_features[resolved].values)
                if resolved != requested:
                    changed += 1
        if missing:
            examples = ", ".join(missing[:10])
            print(
                f"-> WARNING: {len(missing)} {role} were not found in the input dataset. "
                f"Using zero-filled placeholders so reconstruction can continue. "
                f"First missing values: {examples}."
            )
        if changed:
            print(f"-> INFO: Mapped {changed} {role} to current dataset column names.")
        if not values:
            return np.empty((len(df_features), 0), dtype=np.float32)
        return np.column_stack(values)

    def _diagnostic_covariate_file_candidates(self, diagnostic_covariates_path: str, code: str) -> List[str]:
        code_text = str(code).strip().replace("#", ":")
        variants = [
            code_text,
            code_text.replace("__", "_"),
            code_text.replace("_", "__", 1) if "_" in code_text else code_text,
        ]
        if code_text.startswith("DEMAND_"):
            variants.append(code_text[len("DEMAND_"):])
        else:
            variants.append(f"DEMAND_{code_text}")

        ordered = []
        for variant in variants:
            for normalized in (variant, variant.replace("__", "_")):
                path = diagnostic_covariates_path + normalized + ".xlsx"
                if path not in ordered:
                    ordered.append(path)
        return ordered

    def load_diagnostic_covariates(self,diagnostic_covariates_path,code, forecast):
            """ 
            Load the diagnostics covariates for the given code and forecast horizon from the specified 
            path. The covariates are expected to be stored in an Excel file named after the code, 
            with a sheet containing a 'LAG' column that matches the forecast horizon. 
            The 'predictors' column in that sheet should contain a comma-separated list of covariate names 
            to be used for diagnostics. This function reads the Excel file, 
            filters for the relevant forecast horizon, and returns the list of diagnostic covariates.
            Parameters:
                diagnostic_covariates_path: The directory path where the diagnostic covariate Excel files are stored.
                code: The code for which to load the diagnostic covariates (used to determine the filename).
                forecast: The forecast horizon (lag) for which to load the covariates.
            Returns:
                A list of diagnostic covariate names to be used for the specified code and forecast horizon.
            """
            candidates = self._diagnostic_covariate_file_candidates(diagnostic_covariates_path, code)
            selected_path = next((path for path in candidates if os.path.exists(path)), None)
            if selected_path is None:
                raise FileNotFoundError(
                    f"Diagnostic covariates file not found for code '{code}'. "
                    f"Tried: {', '.join(candidates)}. "
                    "Use a code/forecast with BEST_features data."
                )

            diagnostic_covariates_df = pd.read_excel(selected_path, engine='openpyxl')
            matching_rows = diagnostic_covariates_df[diagnostic_covariates_df['LAG'] ==  forecast]
            if matching_rows.empty:
                available_lags = sorted(diagnostic_covariates_df['LAG'].dropna().astype(int).unique().tolist())
                raise ValueError(
                    f"No diagnostic covariates row found in {selected_path} "
                    f"for LAG={forecast}. Available LAG values: {available_lags}."
                )
            predictors_value = matching_rows['predictors'].iloc[0]
            diagnostic_covariates_list = [
                item.strip()
                for item in str(predictors_value).split(',')
                if item.strip()
            ]
        
            return diagnostic_covariates_list

    def prepare_prediction_univ_data(self, data_path: str, code: str, lookback: int, forecast: int, cutoff_date: str, max_date: str, scaler: FunctionTransformer, eliminate_covid_data: bool, covid_token: bool, production_mode: bool = False, covid_dates: Optional[List[str]] = None):
        """
        Prepare the data for univariate time series forecasting in production. This function loads the data from
        the specified path, processes it to create time series features, and prepares the input (X) and target (Y) arrays for the model.
        It also handles the inclusion of a COVID token feature and the elimination of COVID-affected data if specified. 
        The function returns the prepared input and target arrays, as well as the corresponding timestamps.
        Parameters:
            data_path: Path to the input data file (Parquet format).
            code: The target variable code for which the forecast is to be made.
            lookback: The number of past time steps to include in the input features.
            forecast: The number of future time steps to predict.
            cutoff_date: The date to use as the cutoff for training data.
            max_date: The maximum date to include in the data.
            scaler: A FunctionTransformer object for scaling the features.
            eliminate_covid_data: A boolean indicating whether to eliminate data affected by COVID-19.
            covid_token: A boolean indicating whether to include a COVID token feature.
            production_mode: A boolean indicating whether the function is being run in production mode (default is False).
            covid_dates: An optional list of date strings representing the COVID-affected periods to be eliminated if eliminate_covid_data is True.
        Returns:
            X_raw: A numpy array containing the input features for the model.
            Y_raw: A numpy array containing the target values for the model.
            df_timestamp: A pandas Series containing the timestamps corresponding to the input features and target valuesç
        """
        # Load CSV or Parquet
        df = smart_read(data_path)
        if eliminate_covid_data:
            assert covid_dates is not None
            df = data_preparation.eliminate_covid_dates(df, covid_dates)
        
        df = data_preparation.cut_dataframe(df, cutoff_date,max_date, data_path)

        df_timestamp = df['timestamp']
         
        # Match the training univariate contract: target series plus optional COVID token.
        target_col = self._resolve_column(df.columns, code, "target code")
        
        X_raw = df[target_col].values.reshape(-1, 1).astype(np.float32)
        Y_raw = df[target_col].values.astype(np.float32)
        
        if covid_token:
            df_covid = data_preparation.add_covid_token(df)
            covid_feature = df_covid['covid_token'].values.reshape(-1, 1).astype(np.float32)
            X_raw = np.hstack((X_raw, covid_feature))

        if production_mode:
            X_raw = X_raw[-lookback:].reshape(1, lookback, -1)
            Y_raw = Y_raw[-forecast:].reshape(1, forecast)
            Y_raw = np.zeros_like(Y_raw)  # Replace with zeros for production mode as we don't have true future values
            
            # Generate future timestamps for the forecast horizon and append to df_timestamp
            last_date = df_timestamp.iloc[-1]
            new_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1), 
                periods=forecast, 
                freq='D'
            )
            df_forecast = pd.Series(new_dates)
            #df_timestamp = df_timestamp.iloc[-lookback:].reset_index(drop=True)
            df_timestamp = pd.concat([df_timestamp, df_forecast], ignore_index=True)
            

        else:
            X_raw = X_raw[-(lookback + forecast):-forecast].reshape(1, lookback, -1)
            Y_raw = Y_raw[-forecast:].reshape(1, forecast)
            #df_timestamp = df_timestamp#.iloc[-(lookback + forecast):].reset_index(drop=True)
        return X_raw, Y_raw, df_timestamp


    def prepare_prediction_diagnostics_data(self, data_path: str, code: str, lookback: int, forecast: int,cutoff_date: str, max_date: str, scaler: FunctionTransformer, covid_token: bool = False, production_mode: bool = False, covid_dates: Optional[List[str]] = None, eliminate_covid_data: bool = False): 
        """ 
        Prepares diagnostic data for prediction. This function loads the data, processes it to create time series features, and prepares the input (X) and target (Y) arrays for diagnostics. It also handles the inclusion of a COVID token feature and the elimination of COVID-affected data if specified. The function returns the prepared input and target arrays for diagnostics.
        Parameters:
            data_path: Path to the input data file (Parquet format).
            code: The target variable code for which the diagnostics are to be made.
            lookback: The number of past time steps to include in the input features.
            forecast: The number of future time steps to predict.
            cutoff_date: The date to use as the cutoff for training data.
            max_date: The maximum date to include in the data.
            scaler: A FunctionTransformer object for scaling the features.
            covid_token: A boolean indicating whether to include a COVID token feature (default is False).
            production_mode: A boolean indicating whether the function is being run in production mode (default is False).
            covid_dates: An optional list of date strings representing the COVID-affected periods to be eliminated if eliminate_covid_data is True.
            eliminate_covid_data: A boolean indicating whether to eliminate data affected by COVID-19 (default is False).
        Returns:
            X_raw: A numpy array containing the input features for diagnostics.
            Y_raw: A numpy array containing the target values for diagnostics.
        """
        relevant_feature_cols = self.load_diagnostic_covariates(self.config.diagnostic_covariates_path, code, forecast)
        # Load CSV or Parquet
        df = smart_read(data_path)
        if eliminate_covid_data:
            assert covid_dates is not None
            df = data_preparation.eliminate_covid_dates(df, covid_dates)
        df = data_preparation.cut_dataframe(df, cutoff_date,max_date, data_path)

        # Match the training diagnostics contract: selected diagnostic predictors plus optional COVID token.
        target_col = self._resolve_column(df.columns, code, "target code")
        df_features = df.drop(columns = ['timestamp'])

        if relevant_feature_cols is not None:
            X_raw = self._resolve_feature_values(
                df_features,
                relevant_feature_cols,
                "diagnostic predictor columns",
            ).astype(np.float32)
        else:
            X_raw = df_features.values.astype(np.float32)

        Y_raw = df[target_col].values.astype(np.float32)
        if covid_token:
            df_covid = data_preparation.add_covid_token(df)
            covid_feature = df_covid['covid_token'].values.reshape(-1, 1).astype(np.float32)
            
            X_raw = np.hstack((X_raw, covid_feature))

        if production_mode:
            X_raw = X_raw[-lookback:].reshape(1, lookback, -1)
            Y_raw = Y_raw[-forecast:].reshape(1, forecast)
            Y_raw = np.zeros_like(Y_raw)  # Replace with zeros for production mode as we don't have true future values
        else:
            X_raw = X_raw[-(lookback + forecast):-forecast].reshape(1, lookback, -1)
            Y_raw = Y_raw[-forecast:].reshape(1, forecast)
        return X_raw, Y_raw
    

    def prepare_prediction_seasonal_data(self, data_path: str, code: str, forecast: int, lookback: int, cutoff_date: str, max_date: str, categorical_vars: List[str], predictions_train: Optional[np.ndarray], predictions_test: Optional[np.ndarray], scaler: FunctionTransformer):
        """
        Prepares seasonal data for prediction. This function loads the data, processes it to create time series features, and prepares the input (X) array for seasonal covariates. The function returns the prepared input array for seasonal covariates.
        Parameters:
            data_path: Path to the input data file (Parquet format).
            code: The target variable code for which the seasonal covariates are to be prepared.
            forecast: The number of future time steps to predict.
            lookback: The number of past time steps to include in the input features.
            cutoff_date: The date to use as the cutoff for training data.
            max_date: The maximum date to include in the data.
            categorical_vars: A list of categorical variable names to be included in the features.
            predictions_train: An optional numpy array containing the model's predictions on the training data, which can be used to create lag features for the seasonal covariates.
            predictions_test: An optional numpy array containing the model's predictions on the test data, which can be used to create lag features for the seasonal covariates.
            scaler: A FunctionTransformer object for scaling the features.
        Returns:
            X_seasonal_covs: A numpy array containing the input features for the seasonal covariates, prepared for the specified code and forecast horizon.
        """
        df = smart_read(data_path)
        # Prepare seasonal features for training data

        print("Preparing seasonal features for training data...")
        df_processed = data_preparation.prepare_time_series_features(
            df, 
            categorical_vars, 
            cutoff_date=cutoff_date,
            max_date = max_date,
            scaler = scaler,
            eliminate_covid_data=self.config.eliminate_covid_data, 
            covid_dates=self.config.covid_dates,)

        X_seasonal_covs = df_processed.drop(columns=['timestamp']).values[-forecast:].reshape(1, -1, df_processed.shape[1]-1)
        X_seasonal_covs = X_seasonal_covs.astype(float)


        return X_seasonal_covs
