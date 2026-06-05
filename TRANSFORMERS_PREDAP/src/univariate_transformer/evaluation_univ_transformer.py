
import os
import sys 
from src.utils.mlflow_logger import MLflowLogger
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import pandas as pd
import numpy as np
import time


# Add the src directory to path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir) if os.path.basename(current_dir) != 'src' else current_dir
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from utils import data_preparation
from univariate_transformer import plt_model, plot_predictions_with_waves, extract_model_params
#from residual_multivariate_transformers import load_base_model_transformer


def evaluate_univ_transformer(model_name, input_directory, code,  cutoff_date, max_date, covid_token = False, MODEL_FOLDER='models_univariate_transformer', df_waves=None, scaler = None, eliminate_covid_data=False, covid_dates=None, relevant_feature_cols=None, batch_size = 32, split_ratio=None):
    print(f"\n--- Evaluating model: {model_name} ---")
        
    # Extract parameters from filename
    code = code.replace("#", ":")
    lookback, forecast = extract_model_params(model_name)
    
    if lookback is None or forecast is None:
        print(f"Skipping {model_name} - could not extract parameters, the lookback or forecast value is None.")
        return  # Stop evaluation for this model

    # Load model
    model_path = os.path.join(MODEL_FOLDER, model_name)
    model = tf.keras.models.load_model(model_path, compile=True)


    # Prepare test data
    X_test, Y_test = data_preparation.prepare_data(
        input_directory, 
        code, 
        lookback, 
        forecast,covid_token=covid_token,  
        cutoff_date=cutoff_date,
        max_date = max_date,
        train=False, 
        debug=True, 
        univariate=True, 
        scaler=scaler, 
        eliminate_covid_data=eliminate_covid_data, 
        covid_dates=covid_dates, 
        relevant_feature_cols=relevant_feature_cols, 
        split_ratio=split_ratio
    )

    X_test_orig, Y_test_orig = data_preparation.prepare_data_not_normalized(
        input_directory, 
        code, 
        lookback, 
        forecast,
        covid_token=covid_token,  
        cutoff_date=cutoff_date, 
        max_date = max_date, 
        train=False, 
        debug=True, 
        univariate=True, 
        eliminate_covid_data=eliminate_covid_data, 
        covid_dates=covid_dates, 
        relevant_feature_cols=relevant_feature_cols,
        split_ratio=split_ratio
    )



    date_list = data_preparation.extract_dates(input_directory, 
                                               code, 
                                               lookback,
                                               forecast, 
                                               train=False, 
                                               cutoff_date=cutoff_date, 
                                               max_date = max_date, 
                                               eliminate_covid_data=eliminate_covid_data, 
                                               covid_dates=covid_dates)

    original_scale_df = pd.read_csv(input_directory)
    # Get predictions
    predictions = model.predict(X_test, verbose=0)

    print("Predicted values shape:", predictions.shape)
    
    # Inverse transform predictions
    predictions_to_plot = data_preparation.inverse_transform_predictions(
        predictions, original_scale_df, code=code, forecast=forecast, lookback=lookback, 
        cutoff_date=cutoff_date, max_date = max_date, scaler=scaler, 
        eliminate_covid_data=eliminate_covid_data, covid_dates=covid_dates,
        split_ratio=split_ratio
    )
    
    # Evaluate model
    
    loss, mae, mse = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)

    non_negative_predictions_to_plot = np.maximum(predictions_to_plot, 0)  # Ensure no negative predictions

    original_mae = mean_absolute_error(Y_test_orig, non_negative_predictions_to_plot)
    original_mse = mean_squared_error(Y_test_orig, non_negative_predictions_to_plot)
    original_rmse = np.sqrt(original_mse)
    original_wape = np.sum(np.abs(Y_test_orig - non_negative_predictions_to_plot)) / np.sum(np.abs(Y_test_orig)) * 100
    print(f"Test Results - Loss: {loss:.4f}, MAE: {original_mae:.4f}, MSE: {original_mse:.4f}, RMSE: {original_rmse:.4f}, WAPE: {original_wape:.4f}%")
    

    MLflowLogger(active=True).log_metrics({
        "eval/univ_transformer_loss": loss,
        "eval/univ_transformer_mae": original_mae,
        "eval/univ_transformer_mse": original_mse,
        "eval/univ_transformer_rmse": original_rmse,
        "eval/univ_transformer_wape": original_wape
    })


    # Generate plots
    model_display_name = model_name.replace('.keras', '')


    plt_model(Y_test_orig, predictions_to_plot, date_list, model_name=model_display_name, show_plt=False)
    # Plot predictions with pandemic waves
    plot_predictions_with_waves(Y_test_orig, predictions_to_plot, date_list, df_waves, model_display_name)


    # Sliding window evaluation (optional)
    # evaluate_model_sliding_window(model, model_display_name, X_test, Y_test, date_list, df_waves, sliding_window=forecast)
    return predictions, loss, original_mae, original_mse, original_rmse, original_wape

