

import keras
import mlflow
from sklearn.preprocessing import FunctionTransformer
import tensorflow as tf
import numpy as np
import os
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_architechture.model_architecture_univ_transformer import (
    build_base_model,
    PositionalEncoding as UnivPositionalEncoding,
    RevIN as UnivRevIN,
)

from model_architechture.model_architecture_residual_transformer import (
    hybrid_lstm_transformer_model,
    PositionalEncoding as ResidualPositionalEncoding,
    RevIN as ResidualRevIN,
    CustomCosineDecay as ResidualCustomCosineDecay,

)


import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from config.base_transformer_config import BaseTransformerConfig
from production.data_preparation_in_poduction import DataPreparationInProduction

scaler = FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)


CUSTOM_OBJECTS_UNIV = {
    "UnivariateTransformer": build_base_model,
    'RevIN': UnivRevIN,
    'PositionalEncoding': UnivPositionalEncoding,

}   

CUSTOM_OBJECTS_RESIDUAL = {
    "ResidualTransformer": hybrid_lstm_transformer_model,
    "RevIN": ResidualRevIN,
    "PositionalEncoding": ResidualPositionalEncoding,
    "CustomCosineDecay": ResidualCustomCosineDecay,
}

class ModelQuantizationPipeline(DataPreparationInProduction):
    """Pipeline for loading, quantizing, and evaluating Keras models in a production setting."""
    
    def __init__(self, config: BaseTransformerConfig):
        self.config = config
        self.config.print_config()


    def load_mlflow_model(self, run_id, model_name_in_run, custom_objects=None):
        """Load a Keras model from an specified mlflow experiment runID and model path.

        Args:            
            run_id (str): The mlflow run ID where the model is stored.
            model_name_in_run (str): The path to the model artifact within the mlflow run.
            custom_objects (dict, optional): A dictionary of custom objects (e.g., custom layers) used in the
        Returns:
            keras.Model: The loaded Keras model.                
        """

        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=model_name_in_run)
        actual_model_path = os.path.join(local_path, "data", "model.keras")
        #model_uri = f"runs:/{run_id}/{model_name_in_run}"
        
        # Load the model with safe_mode=False to handle custom layers properly
        with tf.keras.utils.custom_object_scope(custom_objects or {}):
            model = tf.keras.models.load_model(
                actual_model_path,
                compile=False,
                safe_mode=False
            )
        
        return model

    def manual_weight_quantization(self, model, model_name,  output_path="_f16.keras"):
        """Manually quantize the weights of a Keras model to float16 precision.
        Args:
            model (keras.Model): The original Keras model to be quantized.
            model_name (str): A name identifier for the model, used for saving the quantized weights.
            output_path (str): The suffix for the saved quantized model file.
        Returns:
            keras.Model: A new Keras model instance with quantized weights.
        """
        # 1. Create a deep copy so we don't modify the original model
        quant_model = keras.models.clone_model(model)
        quant_model.set_weights(model.get_weights())
        
        for layer in quant_model.layers:
            weights = layer.get_weights()
            if weights:
                # 2. Actually use float16 for compression
                quantized_weights = [w.astype(np.float16) for w in weights]

                layer.set_weights(quantized_weights)
        #save_weights_path =  model_name + "_f16_weights.h5"
        #quant_model.save_weights(save_weights_path)
                
        #quant_model.save(model_name + output_path)
        return quant_model


    def evaluate_model(self, model,model_name, past_preds,  X_test, Y_test):
        """Evaluate the model on the test set and return performance metrics.
        Args:
            model (keras.Model): The Keras model to evaluate.
            model_name (str): A name identifier for the model, used for logging purposes.
            past_preds (np.array or None): If evaluating a residual model, the predictions from the previous model in the pipeline to be added to the current predictions for a fair comparison. For the univariate model, this should be None.
            X_test (np.array): The test features.
            Y_test (np.array): The true target values for the test set.
        Returns:
            tuple: A tuple containing the MAE, MSE, RMSE, WAPE, and the raw predictions from the model (before adding past_preds).
        """
        # Get predictions
        predictions = model.predict(X_test, verbose=0)
        if past_preds is not None:
            corrected_preds =  past_preds + predictions
        else:
            corrected_preds = predictions

        print("Predicted values shape:", predictions.shape)

        # Inverse transform predictions
        '''predictions_to_plot = data_preparation.inverse_transform_predictions(
            corrected_preds, original_scale_df, code=code, forecast=forecast, lookback=lookback, cutoff_date=cutoff_date, max_date = max_date, scaler=scaler, eliminate_covid_data=eliminate_covid_data, covid_dates=covid_dates
        )'''

        non_negative_predictions_to_plot = np.maximum(corrected_preds, 0)  # Ensure no negative predictions

        original_mae = mean_absolute_error(Y_test, non_negative_predictions_to_plot)
        original_mse = mean_squared_error(Y_test, non_negative_predictions_to_plot)
        original_rmse = np.sqrt(original_mse)
        original_wape = np.sum(np.abs(Y_test - non_negative_predictions_to_plot)) / np.sum(np.abs(Y_test)) * 100
        
        print(f"Test Results for {model_name} - MAE: {original_mae:.4f}, MSE: {original_mse:.4f}, RMSE: {original_rmse:.4f}, WAPE: {original_wape:.4f}%")

        return original_mae, original_mse, original_rmse, original_wape, corrected_preds


    def save_quantized_model_weights(self, quant_model, model_name, code, forecast, lookback):
        """Save the weights of the quantized model to a specified directory with a structured naming convention.
        Args:
            quant_model (keras.Model): The quantized Keras model whose weights are to be saved.
            model_name (str): A name identifier for the model, used for naming the saved weights file.
            code (str): The code identifier for the dataset/model, used in the naming convention.
            forecast (int): The forecast horizon, used in the naming convention.
            lookback (int): The lookback period, used in the naming convention.
        Returns:
            None: The function saves the weights to disk and does not return any value."""
        save_weights_path =  f"../quantized_models/{code}/{model_name}/{code}_{model_name}_{forecast}fh_{lookback}lb_f16_weights.h5"
        if not os.path.exists(os.path.dirname(save_weights_path)):
            os.makedirs(os.path.dirname(save_weights_path))
        quant_model.save_weights(save_weights_path)
        print(f"Quantized weights saved to {save_weights_path}")

    def load_mlflow_run_id_by_name(self, exp_names, code, forecast, lookback, model_type, lr = 1e-5):
        """Load the mlflow run ID for a given model configuration based on a structured naming convention.
        Args:
            exp_names (list of str): A list of mlflow experiment names to search within.
            code (str): The code identifier for the dataset/model, used in the naming convention.
            forecast (int): The forecast horizon, used in the naming convention.
            lookback (int): The lookback period, used in the naming convention.
            model_type (str): The type of model (e.g., "univariate_model", "diagnostics_model", "seasonal_model"), used in the naming convention.
            lr (float, optional): The learning rate used in the model training, used in the naming convention. Default is 1e-5.
        Returns:
            str: The mlflow run ID that matches the specified configuration.
        Raises:
            ValueError: If no run is found that matches the specified configuration.
        """

        #run_name_prefix = f"full_TRANSFORMER3_transformer_{code}_lb{lookback}_fh{forecast}"
        run_name_prefix = f"1.0_Production_TRANSFORMER_{code}_lb{lookback}_fh{forecast}"
        filter_string = f"attributes.run_name LIKE '{run_name_prefix}%'"
        
        runs = mlflow.search_runs(experiment_names = exp_names,
                                filter_string=filter_string, 
                                order_by=["start_time DESC"],
                                max_results=1)
        
        if runs.empty:
            raise ValueError(f"No run found starting with '{run_name_prefix}' in '{exp_names}'")
        
        return runs.iloc[0].run_id
    
    def eval_quantization_impact(self, input_directory, code, lookback, forecast,cutoff_date, 
                                 max_date,scaler, univ_model, diagnostics_model, seasonal_model, 
                                 quant_univ_model, quant_diagnostics_model, quant_seasonal_model):
        """Evaluate the impact of quantization on model performance by comparing the original and quantized models on the same test set.
        Args:
            input_directory (str): The directory path where the test data is stored.
            code (str): The code identifier for the dataset/model, used in data preparation.
            lookback (int): The lookback period, used in data preparation.
            forecast (int): The forecast horizon, used in data preparation.
            cutoff_date (datetime): The cutoff date for the test data.
            max_date (datetime): The maximum date for the test data.
            scaler (object): The scaler object used for data normalization.
            univ_model (object): The original univariate model.
            diagnostics_model (object): The original diagnostics model.
            seasonal_model (object): The original seasonal model.
            quant_univ_model (object): The quantized univariate model.
            quant_diagnostics_model (object): The quantized diagnostics model.
            quant_seasonal_model (object): The quantized seasonal model.
        Returns:
            None: The function prints the evaluation results and does not return any value.
        """
        
        # --- 1. DATA PREPARATION PHASE ---
        # Load all required test features
        original_scale_df = pd.read_parquet(input_directory)

        X_test, Y_test, df_timestamp = self.prepare_prediction_univ_data(
            data_path=input_directory, 
            code=code, 
            lookback=lookback, 
            forecast=forecast, 
            cutoff_date=cutoff_date, 
            max_date=max_date, 
            scaler=scaler, 
            eliminate_covid_data=eliminate_covid_data, 
            covid_token=self.config.covid_token,
            production_mode=False,
            covid_dates=covid_dates
        )
        
       
        X_test_covs, Y_test_covs = self.prepare_prediction_diagnostics_data(
            data_path=input_directory, 
            code=code, 
            lookback=lookback, 
            forecast=forecast, 
            cutoff_date=cutoff_date, 
            max_date=max_date, 
            scaler=scaler, 
            covid_token=self.config.covid_token,
            production_mode=False,
            covid_dates=covid_dates,
            eliminate_covid_data=eliminate_covid_data
        )


        categorical_vars = ["Day_of_Week", "Month", "Season", "Holiday", "School_Vacation", "Is_Weekend"]#


        # --- 2. INFERENCE PHASE ---
        predictions_univ = univ_model.predict(X_test, verbose=0)
        pred_diagnostics_residuals = diagnostics_model.predict(X_test_covs, verbose=0)

        predictions_test = predictions_univ + pred_diagnostics_residuals 

        X_test_seasonal_covs = self.prepare_prediction_seasonal_data(
            data_path=input_directory, 
            code=code, 
            lookback=lookback, 
            forecast=forecast, 
            cutoff_date=cutoff_date, 
            max_date=max_date, 
            categorical_vars=categorical_vars, 
            predictions_train=None, 
            predictions_test=predictions_test, 
            scaler=scaler, 
        )
        X_test_seasonal_covs = np.concatenate([X_test_seasonal_covs, predictions_test[:,:,np.newaxis]], axis=-1) 
        X_test_seasonal_covs = X_test_seasonal_covs.astype(np.float32)

        pred_seasonal_residuals = seasonal_model.predict(X_test_seasonal_covs, verbose=0)

        pred_corrected_diagnostics = predictions_univ + pred_diagnostics_residuals
        pred_corrected_seasonal = pred_corrected_diagnostics + pred_seasonal_residuals

        non_negative_predictions_to_plot = np.maximum(pred_corrected_seasonal, 0)  # Ensure no negative predictions

        original_mae = mean_absolute_error(Y_test, non_negative_predictions_to_plot)
        original_mse = mean_squared_error(Y_test, non_negative_predictions_to_plot)
        original_rmse = np.sqrt(original_mse)
        original_wape = np.sum(np.abs(Y_test - non_negative_predictions_to_plot)) / np.sum(np.abs(Y_test)) * 100

        # Evaluate model
        print(f"\nTest Results for Original Models")
        original_mae, original_mse, original_rmse, original_wape, preds_univ = self.evaluate_model(univ_model, "Univariate Transformer", None, X_test, Y_test, )
        diagnostics_mae, diagnostics_mse, diagnostics_rmse, diagnostics_wape, preds_diagnostics = self.evaluate_model(diagnostics_model, "Diagnostics Residuals Transformer", preds_univ, X_test_covs, Y_test, )
        seasonal_mae, seasonal_mse, seasonal_rmse, seasonal_wape, preds_seasonal = self.evaluate_model(seasonal_model, "Seasonal Residuals Transformer", preds_diagnostics , X_test_seasonal_covs, Y_test,)

        #Evaluate quantized model
        quant_predictions_univ = quant_univ_model.predict(X_test, verbose=0)
        quant_predictions_diagnostics_residuals = quant_diagnostics_model.predict(X_test_covs, verbose=0)
        quant_predictions_seasonal_residuals = quant_seasonal_model.predict(X_test_seasonal_covs, verbose=0)

        quant_pred_corrected_diagnostics = quant_predictions_univ + quant_predictions_diagnostics_residuals
        quant_pred_corrected_seasonal = quant_pred_corrected_diagnostics + quant_predictions_seasonal_residuals
        print(f"\nTest Results for Quantized Models")
        original_quant_mae, original_quant_mse, original_quant_rmse, original_quant_wape, _ = self.evaluate_model(quant_univ_model, "Quantized Univariate Transformer", None, X_test, Y_test, )
        diagnostics_quant_mae, diagnostics_quant_mse, diagnostics_quant_rmse, diagnostics_quant_wape, _ = self.evaluate_model(quant_diagnostics_model, "Quantized Diagnostics Residuals Transformer", quant_predictions_univ, X_test_covs, Y_test, )
        seasonal_quant_mae, seasonal_quant_mse, seasonal_quant_rmse, seasonal_quant_wape, _ = self.evaluate_model(quant_seasonal_model, "Quantized Seasonal Residuals Transformer", quant_pred_corrected_diagnostics, X_test_seasonal_covs, Y_test,)

        print("\nComplete vs quantized model comparison:")
        print(f"Original MAE: {original_mae:.4f}, Quantized MAE: {original_quant_mae:.4f}")
        print(f"Original MSE: {original_mse:.4f}, Quantized MSE: {original_quant_mse:.4f}")
        print(f"Original RMSE: {original_rmse:.4f}, Quantized RMSE: {original_quant_rmse:.4f}")
        print(f"Original WAPE: {seasonal_wape:.4f}%, Quantized WAPE: {original_quant_wape:.4f}%")
        print("================================================================\n")
    
    def run_quantization_pipeline(self,exp_names, input_directory, code, lookback, forecast, cutoff_date, max_date, scaler, eliminate_covid_data=False, covid_dates=None):
        """Run the full model quantization pipeline: load models, quantize weights, save quantized weights, and evaluate performance impact.
        Args:
            exp_names (list of str): A list of mlflow experiment names to search within for model loading.
            input_directory (str): The directory path where the test data is stored.
            code (str): The code identifier for the dataset/model, used in data preparation and model loading.
            lookback (int): The lookback period, used in data preparation and model loading.
            forecast (int): The forecast horizon, used in data preparation and model loading.
            cutoff_date (datetime): The cutoff date for the test data.
            max_date (datetime): The maximum date for the test data.
            scaler (object): The scaler object used for data normalization. 
            eliminate_covid_data (bool, optional): Whether to eliminate COVID-19 data from the test set. Default is False.
            covid_dates (list of str, optional): A list of date strings representing COVID-19 periods to eliminate if eliminate_covid_data is True. Default is None.
        Returns:
            tuple: A tuple containing the original univariate model, diagnostics model, seasonal model, and their quantized counterparts.
        """
        univ_model_name_in_run = "univariate_model"
        diag_model_name_in_run = "residual_diagnostics_model"
        seasonal_model_name_in_run = "residual_seasonal_model"

        run_id = self.load_mlflow_run_id_by_name(exp_names=exp_names, code=code, forecast=forecast, lookback=lookback, model_type=None, lr=1e-5)

        univ_model = self.load_mlflow_model(run_id, univ_model_name_in_run, custom_objects= CUSTOM_OBJECTS_UNIV) 
        diagnostics_model = self.load_mlflow_model(run_id, diag_model_name_in_run, custom_objects= CUSTOM_OBJECTS_RESIDUAL)
        seasonal_model = self.load_mlflow_model(run_id, seasonal_model_name_in_run, custom_objects= CUSTOM_OBJECTS_RESIDUAL)
            

        quant_univ_model = self.manual_weight_quantization(univ_model, model_name="univariate_model")
        quant_diagnostics_model = self.manual_weight_quantization(diagnostics_model, model_name="diagnostics_model")
        quant_seasonal_model = self.manual_weight_quantization(seasonal_model, model_name="seasonal_model")

        self.save_quantized_model_weights(quant_univ_model, "univariate_model", code, forecast, lookback)
        self.save_quantized_model_weights(quant_diagnostics_model, "diagnostics_model", code, forecast, lookback)
        self.save_quantized_model_weights(quant_seasonal_model, "seasonal_model", code, forecast, lookback)

        return univ_model, diagnostics_model, seasonal_model, quant_univ_model, quant_diagnostics_model, quant_seasonal_model



if __name__ == "__main__":
    CODES_LIST = ["demanda__SERVEI_CODI__INF",
                    "demanda__SERVEI_CODI__INFP",
                    "demanda__SERVEI_CODI__MF",
                    "demanda__SERVEI_CODI__PED",
                    "demanda__SERVEI_CODI__URG",
                    "demanda__TIPUS_CLASS__9T",
                    "demanda__TIPUS_CLASS__C9C",
                    "demanda__TIPUS_CLASS__C9R",
                    "demanda__TIPUS_CLASS__CALTRE",
                    "demanda__TIPUS_CLASS__D9D",
                    "demanda__TIPUS_CLASS__DALTRE"
                    ]#['demanda__TOTAL', 'demanda__SERVEI_CODI__URG', 'B34','J00', 'I10', 'M54','Ch01#subch01#A00-A09']
    
    LOOKBACK_LIST = [7, 14, 60, 60, 182,182]
    FORECAST_LIST = [7, 14, 30, 60, 182,365]


    input_directory = '../data/FINAL_DB/finals_combined.csv'
    models_directory = '../transformer_outputs/models_covid_token'
    scaler = FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)
    max_date = '2027-09-30'
    cutoff_date = '2008-01-01'
    eliminate_covid_data = False
    covid_dates = None

    for code in CODES_LIST:
        for lookback, forecast in zip(LOOKBACK_LIST, FORECAST_LIST):
            print(f"Processing code {code} with lookback {lookback} and forecast {forecast}")
            default_config = BaseTransformerConfig()

            run_quantization_pipeline = ModelQuantizationPipeline(config=default_config)
            #exp_names = ["full_TRANSFORMER3_EXPERIMENTS_TRANSFORMERS_PREDAP_HYDRA_GRID_SEARCH_20260210", "full_TRANSFORMER3_TRANSFORMERS_PREDAP_HYDRA_GRID_SEARCH_20260212"]
            exp_names = ['1.0_Production_TRANSFORMER_TRANSFORMERS_PREDAP_HYDRA_GRID_SEARCH_20260514']
            univ_model, diagnostics_model, seasonal_model, quant_univ_model, quant_diagnostics_model, quant_seasonal_model = run_quantization_pipeline.run_quantization_pipeline(
                exp_names=exp_names,
                input_directory=input_directory,
                code=code,
                lookback=lookback,
                forecast=forecast,
                cutoff_date=cutoff_date,
                max_date=max_date,
                scaler=scaler,
                eliminate_covid_data=eliminate_covid_data,
                covid_dates=covid_dates
            )
            run_quantization_pipeline.eval_quantization_impact(
                input_directory=input_directory,
                code=code,
                lookback=lookback,
                forecast=forecast,
                cutoff_date=cutoff_date,
                max_date=max_date,
                scaler=scaler,
                univ_model=univ_model,
                diagnostics_model=diagnostics_model,
                seasonal_model=seasonal_model,
                quant_univ_model=quant_univ_model,
                quant_diagnostics_model=quant_diagnostics_model,
                quant_seasonal_model=quant_seasonal_model,
            )

            
