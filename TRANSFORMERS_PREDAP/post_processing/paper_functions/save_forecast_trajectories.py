from tensorflow import keras
import os 
from model_architechture.model_architecture_univ_transformer import (
    PositionalEncoding, RevIN
)

from src.config.base_transformer_config import BaseTransformerConfig
from data_utils import data_preparation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
from src.utils.experiments_utils import smart_read
import matplotlib.dates as mdates
import mlflow

from data_utils.data_preparation import (
    split_train_test
    )

from src.training.training_residual_transformer import (
        filter_diagnostics_covariates,
    )

def load_diagnostic_covariates(code, forecast):
        diagnostic_covariates_path = f'../data/best_features/BEST_features_NOSMOOTH_{code}.xlsx'
        diagnostic_covariates_df = pd.read_excel(diagnostic_covariates_path, engine='openpyxl')
        diagnostic_covariates_list = list(diagnostic_covariates_df[diagnostic_covariates_df['LAG'] == forecast]['predictors'])[0].split(',')
    
        return diagnostic_covariates_list

# 2. Reemplazar la función original con nuestra función "inteligente"
pd.read_csv = smart_read

def plt_model(y_test_inverse, yhat_inverse, date_list, model_name, ci=1.96, show_plt=False):
    """
    Plot model results comparing true vs predicted values across a forecast horizon,
    showing mean and confidence regions computed from multiple prediction samples.

    Parameters:
    -----------
    y_test_inverse : array-like, shape (num_of_predictions, forecast_horizon)
        True values (inverse transformed) for each prediction sample.
    yhat_inverse : array-like, shape (num_of_predictions, forecast_horizon)
        Predicted values (inverse transformed) for each prediction sample.
    model_name : str
        Name of the model for the plot title / filename.
    ci : float
        Multiplier for standard deviation to plot confidence region (default 1.96 ≈ 95%).
    show_plt : bool
        Whether to display the plot interactively.
    """
    try:
        y_test_arr = np.asarray(y_test_inverse)
        yhat_arr = np.asarray(yhat_inverse)

        # Accept 1D inputs by treating them as single-sample predictions
        if y_test_arr.ndim == 1:
            y_test_arr = y_test_arr.reshape(-1, 1)
        if yhat_arr.ndim == 1:
            yhat_arr = yhat_arr.reshape(-1, 1)

        if y_test_arr.shape != yhat_arr.shape:
            raise ValueError("y_test_inverse and yhat_inverse must have the same shape "
                             "(num_of_predictions, forecast_horizon)")

        n_samples, horizon = y_test_arr.shape
        x = np.arange(1, horizon + 1)

        # Compute statistics across prediction samples (axis=0 -> across samples for each horizon step)
        mean_true = np.nanmean(y_test_arr, axis=1)
        std_true = np.nanstd(y_test_arr, axis=1)

        mean_pred = np.nanmean(yhat_arr, axis=1)
        std_pred = np.nanstd(yhat_arr, axis=1)


        # pick middle column (robust to 1D arrays)
        if y_test_arr.ndim > 1:
            middle_value = int(y_test_arr.shape[1] // 2)
        else:
            middle_value = 0
        fig, ax = plt.subplots(figsize=(20, 8))
        # Plot mean lines
        ax.plot(date_list, y_test_arr[:, 3], label='True (mean)', marker='o', linestyle='-', alpha=0.9)
        ax.plot(date_list, yhat_arr[:, 3], label='Predicted (mean)', marker='x', linestyle='--', alpha=0.9)

        # Plot confidence regions: mean ± ci * std
        upper_true = mean_true + ci * std_true
        lower_true = mean_true - ci * std_true
        upper_pred = mean_pred + ci * std_pred
        lower_pred = mean_pred - ci * std_pred

        '''ax.fill_between(date_list, lower_true, upper_true, color='blue', alpha=0.15, label=f'True ± {ci}σ')
        ax.fill_between(date_list, lower_pred, upper_pred, color='orange', alpha=0.15, label=f'Predicted ± {ci}σ')
        '''
        ax.set_xlabel('Forecast Horizon Step', fontweight='bold', fontsize=12)
        ax.set_ylabel('Value', fontweight='bold', fontsize=12)
        ax.set_title(f'Real vs. Predicted (mean ± {ci}σ) // MODEL: {model_name}')
        ax.legend()
        ax.grid(alpha=0.3)

        fig.tight_layout()
        os.makedirs("plots_paper", exist_ok=True)
        fig.savefig(f"plots_paper/model_results_{model_name}_horizon.png")
        mlflow.log_artifact(f"plots_paper/model_results_{model_name}_horizon.png", artifact_path="plots_paper")
        if show_plt:
            plt.show()
        plt.close(fig)

    except Exception as e:
        print(f"Error plotting model results: {str(e)}")

'''def plot_final_forecast(Y_test_orig, predictions_to_plot, date_list):
    """
    Plots the final 365-day window comparing Ground Truth and Predictions.
    
    Shapes expected:
    - Y_test_orig: (750, 365)
    - predictions_to_plot: (750, 365)
    - date_list: (750, 365) or (750,) containing lists of 365 dates
    """
    
    # 1. Extract the last instance (index -1)
    # If date_list is an array of 750 elements where each element is a list of 365 dates:
    actual_values = Y_test_orig[-1]
    predicted_values = predictions_to_plot[-1]
    dates = date_list[-365:] 
    
    # 2. Create the plot
    plt.figure(figsize=(15, 7))
    
    # Plotting both lines
    plt.plot(dates, actual_values, label='Ground Truth (Original Scale)', 
             color='#2c3e50', linewidth=2, alpha=0.8)
    plt.plot(dates, predicted_values, label='Model Prediction', 
             color='#e74c3c', linestyle='--', linewidth=2)
    
    # 3. Formatting the UI
    plt.title("Final 365-Day Forecast Comparison", fontsize=15, pad=20)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    
    # Improve date formatting on X-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2)) # Show label every 2 months
    plt.xticks(rotation=45)
    
    plt.legend(loc='upper left', frameon=True)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("plots_paper/final_365_day_forecast_comparison.png")
    plt.show()'''

def plot_multiple_forecasts(Y_test_orig_list, predictions_to_plot_list, pred_corrected_diagnostics_list, pred_corrected_seasonal_list, date_list_list, codes, wapes, diagnostics_wape_list=None, seasonal_wape_list=None):
    """
    Plots multiple forecast comparisons in subplots.
    
    Args:
        Y_test_orig_list: List of arrays [ (750, 365), ... ]
        predictions_to_plot_list: List of arrays [ (750, 365), ... ]
        date_list_list: List of arrays [ (750, 365), ... ]
        codes: List of strings (e.g., ['M54 Univ', 'M54 Diag', 'M54 Seasonal'])
        wapes: List of WAPE error values (floats)
    """
    num_models = len(Y_test_orig_list)
    
    # Create subplots (one row for each model)
    fig, axes = plt.subplots(nrows=num_models, ncols=1, figsize=(15, 5 * num_models), sharex=False)
    
    # Ensure axes is an array even if there is only one plot
    if num_models == 1:
        axes = [axes]

    for i in range(num_models):
        # Extract the last instance (index -1) for the current model
        actual = Y_test_orig_list[i][-200]
        diagnostics_corrected = pred_corrected_diagnostics_list[i][-200]
        seasonal_corrected = pred_corrected_seasonal_list[i][-200]
        predicted = predictions_to_plot_list[i][-200]
        dates = date_list_list[i][-182:]
        code = codes[i]
        wape = wapes[i]
        diagnostics_wape = diagnostics_wape_list[i] if diagnostics_wape_list is not None else None
        seasonal_wape = seasonal_wape_list[i] if seasonal_wape_list is not None else None

        
        ax = axes[i]
        
        # Plot lines
        ax.plot(dates, actual, label='Ground Truth', color='#2c3e50', linewidth=3)
        ax.plot(dates, predicted, label=f'Univariate Prediction - WAPE:{wape:.2f}%', color='#e74c3c', linestyle='--', linewidth=2, alpha=0.8)

        # Plot lines
        
        ax.plot(dates, diagnostics_corrected, label=f'Diagnostics Prediction - WAPE:{diagnostics_wape:.2f}%', color='green', linestyle='--', linewidth=2, alpha=0.8)
        ax.plot(dates, seasonal_corrected, label=f'Seasonal Prediction - WAPE:{seasonal_wape:.2f}%', color='blue', linestyle='--', linewidth=2, alpha=0.8)
        # Title with Code and WAPE
        #wape = wape/100  # Convert percentage to decimal for display
        ax.set_title(f"Code: {code}", fontsize=16, fontweight='bold', pad=10)
        
        # Formatting
        ax.set_ylabel("Original Scale Value", fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Date formatting for X-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    plt.tight_layout()
    plt.savefig("plots_paper/multiple_forecast_comparisons.png")
    plt.show()



default_config = BaseTransformerConfig()

# Extract parameters from Hydra config
CODE = "J00"
lookback = 182
forecast = 365
head_size = default_config.head_size
num_heads = 8
ff_dim = 512
mlp_units = [512, 256]
activation_function = 'gelu'
covid_token = True
cutoff_date = '2008-01-01'
positional_encoding = default_config.positional_encoding
data_path = default_config.data_path
evaluate_model=default_config.evaluate_model
num_transformer_blocks = default_config.num_transformer_blocks
dropout = default_config.dropout
learning_rate = default_config.learning_rate
scaler = default_config.scaler


# Define the path to your file
ex = os.path.exists('../transformer_outputs/models_covid_token/M54_SEASONAL_RESIDUALS_LEARNING_365fh_512ff_182lb_1e-05initlr.keras')
print (f"File exists: {ex}")
model_name_univ = 'M54_base_transformer_365fh_512ff_182lb_1e-05lr.keras'
model_name_diag = 'M54_DIAGNOSTIC_RESIDUALS_LEARNING_365fh_512ff_182lb_1e-05initlr.keras'
model_name_seasonal = 'M54_SEASONAL_RESIDUALS_LEARNING_365fh_512ff_182lb_1e-05initlr.keras'
model_names = [model_name_univ, model_name_diag, model_name_seasonal]
model_path_univ = '../transformer_outputs/models_covid_token/' + model_name_univ
model_path_diag = '../transformer_outputs/models_covid_token/' + model_name_diag
model_path_seasonal = '../transformer_outputs/models_covid_token/' + model_name_seasonal


model_paths_univ_list = [
    'M54_base_transformer_365fh_512ff_182lb_1e-05lr.keras',
    'J00_base_transformer_365fh_512ff_182lb_1e-05lr.keras',
    'Ch01#subch01#A00-A09_base_transformer_365fh_512ff_182lb_1e-05lr.keras'
]

Y_test_orig_list = []
predictions_to_plot_list = []
pred_diagnostics_residuals_to_plot_list = []
pred_seasonal_residuals_to_plot_list = []
date_list_list = []
original_wape_list = []
diagnostics_wape_list = []
seasonal_wape_list = []

#run_id = "b791d7e5459f402fbf465c61018349c2"
univ_model_name_in_run = "univariate_model"
diag_model_name_in_run = "residual_diagnostics_model"
seasonal_model_name_in_run = "residual_seasonal_model"
#model_phases = ['univariate_model', 'residual_diagnostics_model', 'residual_seasonal_model']

run_id_dict = {
    #'demanda__TOTAL': "b363cfbfd2124e908228327673fef429", }
    #'demanda__SERVEI_CODI__URG' : 'b791d7e5459f402fbf465c61018349c2',}
    #'J00': "bd748b533de649a1a2bc0d7e69e9884f", }
    #'Ch01#subch01#A00-A09': "83ae8cc79e1747afb4a3794391c3c054",}
    #'I10' : "f938c2de01a846c2ae1e4749f46fb868",
    "M54" : "e8b90d46995746a5bcfdf50bf54180c6"}

def load_mlflow_model(run_id, model_name_in_run, custom_objects=None):
    """Load a Keras model from an specified mlflow experiment runID and model path."""

    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=model_name_in_run)
    actual_model_path = os.path.join(local_path, "data", "model.keras")
    #model_uri = f"runs:/{run_id}/{model_name_in_run}"

    # Load the model with safe_mode=False to handle custom layers properly
    model = keras.models.load_model(actual_model_path, custom_objects=custom_objects)
    #model = mlflow.keras.load_model(model_uri, custom_objects=custom_objects, safe_mode = False)
    return model

def univariate_transformer_phase(input_directory, code, lookback, forecast, cutoff_date, max_date,scaler,eliminate_covid_data=False, relevant_feature_cols=None, split_ratio: float = 0.8):
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
                                                covid_dates=None)
    
    

    return X_test, Y_test, Y_test_orig, date_list



def diagnostics_transformer_phase(code, lookback, forecast, final_cutoff_date, scaler, split_ratio: float = 0.8):
    train_split, test_split = split_train_test(
            smart_read(data_path), 
            split_ratio=split_ratio, 
            cutoff_date = cutoff_date,
            max_date = final_cutoff_date,
            scaler =scaler
        )
        
        
    diagnostic_covariates_list = load_diagnostic_covariates( code, forecast, lookback, final_cutoff_date)
    train_split = filter_diagnostics_covariates(train_split, diagnostic_covariates_list)
    test_split = filter_diagnostics_covariates(test_split, diagnostic_covariates_list)
    
    # Generate rolling sequences with covariates for training
    print("Generating sequences with diagnostics covariates for training...")
    X_train_covs, _ = data_preparation.prepare_data(
        data_path, 
        code, 
        lookback, 
        forecast,
        covid_token=covid_token, 
        cutoff_date=cutoff_date,
        max_date = final_cutoff_date,
        relevant_feature_cols=diagnostic_covariates_list, 
        train=True, 
        univariate=False,
        scaler = scaler,
        eliminate_covid_data=False,
        covid_dates=None,
        split_ratio=split_ratio
    )
    
    print(f"Training covariates shape: {X_train_covs.shape}")
    print(f"Expected shape: (num_samples, {lookback}, num_features)")
    
    # Generate rolling sequences for test data
    print("Preparing test data with covariates...")
    X_test_covs, _ = data_preparation.prepare_data(
        data_path, 
        code, 
        lookback, 
        forecast, 
        covid_token=covid_token, 
        cutoff_date=cutoff_date, 
        max_date = final_cutoff_date,
        relevant_feature_cols=diagnostic_covariates_list, 
        train=False, 
        univariate=False,
        scaler = scaler,
        eliminate_covid_data=False,
        covid_dates=None,
        split_ratio=split_ratio
    )
    
    print(f"Test covariates shape: {X_test_covs.shape}")
    return X_train_covs, X_test_covs


def seasonal_transformer_phase(code, forecast, lookback,cutoff_date, final_cutoff_date, categorical_vars,predictions_train, predictions_test, scaler=None, split_ratio: float = 0.8):
    df = smart_read(data_path)
    # Prepare seasonal features for training data
    print("Preparing seasonal features for training data...")
    df_processed = data_preparation.prepare_time_series_features(
        df, 
        categorical_vars, 
        cutoff_date=cutoff_date,
        max_date = final_cutoff_date,
        scaler = scaler,
        eliminate_covid_data=False, 
        covid_dates=None,

    )
    
    # Load and split the original data for covariate extraction
    df_train_processed, df_test_processed = split_train_test(
        df_processed, 
        split_ratio=split_ratio, 
        cutoff_date=cutoff_date,
        max_date = final_cutoff_date,
        scaler = scaler, 

    )
    
    
    # Generate rolling sequences with covariates for training
    print("Generating rolling sequences with seasonal covariates for training...")
    X_train_covs = data_preparation.generate_rolling_sequences_covariates(
        df_train_processed, 
        lookback, 
        forecast, 
        predictions_train,

    )
    
    print(f"Training covariates shape: {X_train_covs.shape}")
    print(f"Expected shape: (num_samples, {lookback}, num_features)")
    
    # Generate rolling sequences for test data
    print("Generating rolling sequences with seasonal covariates for test data...")
    X_test_covs = data_preparation.generate_rolling_sequences_covariates(
        df_test_processed, 
        lookback, 
        forecast, 
        predictions_test,

    )

    print(f"Test covariates shape: {X_test_covs.shape}")

    return X_train_covs, X_test_covs


if __name__ == '__main__':
    CUSTOM_OBJECTS = {'PositionalEncoding': PositionalEncoding, 
                      'RevIN': RevIN}
    for code, run_id in run_id_dict.items():

        # Load the model
        univ_model = load_mlflow_model(run_id, univ_model_name_in_run, custom_objects=CUSTOM_OBJECTS) 
        diagnostics_model = load_mlflow_model(run_id, diag_model_name_in_run, custom_objects=CUSTOM_OBJECTS)
        seasonal_model = load_mlflow_model(run_id, seasonal_model_name_in_run, custom_objects=CUSTOM_OBJECTS)
        
        
        
        print("Model loaded successfully!")



        input_directory = '../data/FINAL_DB/full_CAT1.parquet'
        code = code
        max_date = '2027-09-30'
        eliminate_covid_data = False
        covid_dates = None
        
        
        X_test, Y_test, Y_test_orig, date_list = univariate_transformer_phase(input_directory, code, lookback, forecast, cutoff_date, max_date, relevant_feature_cols=None, scaler=scaler, eliminate_covid_data=eliminate_covid_data)
        X_train_covs, X_test_covs = diagnostics_transformer_phase(code, lookback, forecast, max_date, scaler)
        categorical_vars = ["Day_of_Week", 
                            "Month", 
                            "Season", 
                            "Holiday", 
                            "School_Vacation",
                            "Is_Weekend",
                            ]


        Y_test_orig_list.append(Y_test_orig)
        date_list_list.append(date_list)

        original_scale_df = pd.read_parquet(input_directory)
        # Get predictions
        predictions_univ = univ_model.predict(X_test, verbose=0)
        pred_diagnostics_residuals = diagnostics_model.predict(X_test_covs, verbose=0)

        predictions_test = predictions_univ + pred_diagnostics_residuals
        X_train_seasonal_covs, X_test_seasonal_covs = seasonal_transformer_phase(code, forecast, lookback,cutoff_date, max_date, categorical_vars=categorical_vars, predictions_train=None, predictions_test=None, scaler=scaler)
        X_train_seasonal_covs = X_train_seasonal_covs.astype(float)
        X_test_seasonal_covs = X_test_seasonal_covs.astype(float)
        
        pred_seasonal_residuals = seasonal_model.predict(X_test_seasonal_covs, verbose=0)
        


        print("Predicted values shape:", predictions_univ.shape)

        # Inverse transform predictions
        predictions_to_plot = data_preparation.inverse_transform_predictions(
            predictions_univ, original_scale_df, code=code, forecast=forecast, lookback=lookback, cutoff_date=cutoff_date, max_date = max_date, scaler=scaler, eliminate_covid_data=eliminate_covid_data, covid_dates=covid_dates
        )
        pred_diagnostics_residuals_to_plot = data_preparation.inverse_transform_predictions(
            pred_diagnostics_residuals, original_scale_df, code=code, forecast=forecast, lookback=lookback, cutoff_date=cutoff_date, max_date = max_date, scaler=scaler, eliminate_covid_data=eliminate_covid_data, covid_dates=covid_dates
        )
        pred_seasonal_residuals_to_plot = data_preparation.inverse_transform_predictions(
            pred_seasonal_residuals, original_scale_df, code=code, forecast=forecast, lookback=lookback, cutoff_date=cutoff_date, max_date = max_date, scaler=scaler, eliminate_covid_data=eliminate_covid_data, covid_dates=covid_dates
        )

        pred_corrected_diagnostics = predictions_to_plot + pred_diagnostics_residuals_to_plot
        pred_corrected_seasonal = pred_corrected_diagnostics + pred_seasonal_residuals_to_plot


        predictions_to_plot_list.append(predictions_to_plot)
        pred_diagnostics_residuals_to_plot_list.append(pred_corrected_diagnostics)
        pred_seasonal_residuals_to_plot_list.append(pred_corrected_seasonal)

        # Evaluate model

        loss, mae, mse = univ_model.evaluate(X_test, Y_test, verbose=0, batch_size=256)
        diagnostics_loss, diagnostics_mae, diagnostics_mse = diagnostics_model.evaluate(X_test_covs, Y_test, verbose=0, batch_size=256)
        seasonal_loss, seasonal_mae, seasonal_mse = seasonal_model.evaluate(X_test_seasonal_covs, Y_test, verbose=0, batch_size=256)

        non_negative_predictions_to_plot = np.maximum(predictions_to_plot, 0)  # Ensure no negative predictions

        original_mae = mean_absolute_error(Y_test_orig, non_negative_predictions_to_plot)
        original_mse = mean_squared_error(Y_test_orig, non_negative_predictions_to_plot)
        original_rmse = np.sqrt(original_mse)
        original_wape = np.sum(np.abs(Y_test_orig - non_negative_predictions_to_plot)) / np.sum(np.abs(Y_test_orig)) * 100
        diagnostics_wape = np.sum(np.abs(Y_test_orig - pred_corrected_diagnostics)) / np.sum(np.abs(Y_test_orig)) * 100
        seasonal_wape = np.sum(np.abs(Y_test_orig - pred_corrected_seasonal)) / np.sum(np.abs(Y_test_orig)) * 100
        
        print(f"Test Results - Loss: {loss:.4f}, MAE: {original_mae:.4f}, MSE: {original_mse:.4f}, RMSE: {original_rmse:.4f}, WAPE: {original_wape:.4f}%")
        original_wape_list.append(original_wape)
        diagnostics_wape_list.append(diagnostics_wape)
        seasonal_wape_list.append(seasonal_wape)

        

    # Generate plots
    model_display_name = univ_model_name_in_run.replace('.keras', '')

    plot_multiple_forecasts(Y_test_orig_list, predictions_to_plot_list, pred_diagnostics_residuals_to_plot_list, pred_seasonal_residuals_to_plot_list, date_list_list, codes=[code], wapes=original_wape_list, diagnostics_wape_list=diagnostics_wape_list, seasonal_wape_list=seasonal_wape_list)
    preds = univ_model.predict(X_test)

    plt.plot(preds[0,:])
    plt.savefig('forecast_trajectory.png')  # Save the figure to a file
    plt.show()
    # Check the architecture
    univ_model.summary()

