
import numpy as np
import pandas as pd
from sklearn.pipeline import FunctionTransformer
from typing import List, Optional, Tuple
from data_utils import data_preparation
from config.base_transformer_config import BaseTransformerConfig

default_config = BaseTransformerConfig()



class DataPreparationInProduction:
    def __init__(self, config: BaseTransformerConfig = default_config):
        self.config = config
        self.config.print_config()

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

            diagnostic_covariates_df = pd.read_excel(diagnostic_covariates_path + code + ".xlsx", engine='openpyxl')
            diagnostic_covariates_list = list(diagnostic_covariates_df[diagnostic_covariates_df['LAG'] ==  forecast]['predictors'])[0].split(',')
        
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
        code = code.replace("#", ":")
        # Load CSV
        df = pd.read_csv(data_path)
        if eliminate_covid_data:
            assert covid_dates is not None
            df = data_preparation.eliminate_covid_dates(df, covid_dates)
        
        df = data_preparation.cut_dataframe(df, cutoff_date,max_date, data_path)

        categorical_vars = ["Day_of_Week", "Month", "Season", "Holiday", "School_Vacation","Is_Weekend"]
        df_dates = data_preparation.prepare_time_series_features(df, categorical_vars=categorical_vars, cutoff_date=cutoff_date, max_date=max_date, scaler=scaler, eliminate_covid_data=eliminate_covid_data, covid_dates=covid_dates)
        df_timestamp = df['timestamp']
        df_dates = df_dates.drop(columns=['timestamp']) 
         
        # univariate scenario ...................................................
        # Temporary solution for the name mismatch between the codes in the input data and the codes expected by the reconstruction pipeline and the saved models. Remove the first "DEMAND_" characters if they are present, and replace any "_" with "__" to match the format of the saved models.
        #add DEMAND_ in front of the code if it's not already there, to match the format of the input data
        code = "DEMAND_" + code if not code.startswith("DEMAND_") else code
        code = code.replace("__", "_") if "__" in code else code
        idx_code = df.columns.get_loc(code)
        feature_cols = df.columns[idx_code]  
        target_col = df.columns[idx_code]  # Get the target column 
        
        # Convert to numpy arrays
        X_raw = df[feature_cols].values.reshape(-1, 1)  
        X_raw = np.hstack((X_raw, df_dates.values.astype(np.float32)))
        Y_raw = df[target_col].values # Target values
        
        if covid_token:
            df_covid = data_preparation.add_covid_token(df)
            covid_feature = df_covid['covid_token'].values.reshape(-1, 1)
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
        relevant_feature_cols = self.load_diagnostic_covariates(default_config.diagnostic_covariates_path, code, forecast)
        code = code.replace("#", ":")
        # Load CSV
        df = pd.read_csv(data_path)
        if eliminate_covid_data:
            assert covid_dates is not None
            df = data_preparation.eliminate_covid_dates(df, covid_dates)
        df = data_preparation.cut_dataframe(df, cutoff_date,max_date, data_path)

        categorical_vars = ["Day_of_Week", 
                                "Month", 
                                "Season", 
                                "Holiday", 
                                "School_Vacation",
                                "Is_Weekend",
                                ]
        df_dates = data_preparation.prepare_time_series_features(df, categorical_vars=categorical_vars, cutoff_date=cutoff_date, max_date=max_date, scaler=scaler, eliminate_covid_data=eliminate_covid_data, covid_dates=covid_dates)
        df_dates = df_dates.drop(columns=['timestamp']) 
        
        
        # multivariate scenario ..................................................
        # Select feature columns (exclude timestamp & target)
        # Temporary solution for the name mismatch between the codes in the input data and the codes expected by the reconstruction pipeline and the saved models. Remove the first "DEMAND_" characters if they are present, and replace any "_" with "__" to match the format of the saved models.
        #add DEMAND_ in front of the code if it's not already there, to match the format of the input data
        old_code = code
        code = "DEMAND_" + code if not code.startswith("DEMAND_") else code
        code = code.replace("__", "_") if "__" in code else code

        df.columns = [
            col if col == 'timestamp' else ("DEMAND_" + col if not col.startswith("DEMAND_") else col).replace("__", "_")
            for col in df.columns
        ]
        if code not in df.columns:#Temporary solution for the name mismatch between the codes in the input data and the codes expected by the reconstruction pipeline and the saved models. If the code with "DEMAND_" prefix is not found, try without the prefix.
            code = old_code

        
        idx_code = df.columns.get_loc(code)
        df_features = df.drop(columns = ['timestamp'])
        target_col = df.columns[idx_code]  # Target code column
        # Convert DataFrame to numpy arrays
        if relevant_feature_cols is not None:
            X_raw = df_features[relevant_feature_cols].values  
            X_raw = np.hstack((X_raw, df_dates.values.astype(np.float32)))
        else:
            X_raw = df_features.values

        Y_raw = df[target_col].values
        if covid_token:
            df_covid = data_preparation.add_covid_token(df)
            covid_feature = df_covid['covid_token'].values.reshape(-1, 1)
            
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
        df = pd.read_csv(data_path)
        # Prepare seasonal features for training data

        print("Preparing seasonal features for training data...")
        df_processed = data_preparation.prepare_time_series_features(
            df, 
            self.config.DEFAULT_SEASONAL_CATEGORICAL_VARS, 
            cutoff_date=self.config.cutoff_date,
            max_date = self.config.final_cutoff_date,
            scaler = self.config.scaler,
            eliminate_covid_data=self.config.eliminate_covid_data, 
            covid_dates=self.config.covid_dates,)

        X_seasonal_covs = df_processed.drop(columns=['timestamp']).values[-forecast:].reshape(1, -1, df_processed.shape[1]-1)
        X_seasonal_covs = X_seasonal_covs.astype(float)


        return X_seasonal_covs




