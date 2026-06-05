import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend (no GUI)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import mlflow
import mlflow.tensorflow
import mlflow.keras
from datetime import datetime
import os
import tempfile
import json
import pickle

from src.utils import data_preparation
from src.utils import experiments_utils


"""
CLASS-BASED IMPORTS - IMPROVED ARCHITECTURE
==========================================
Using the new OOP approach from main_train_univ_transformer_class.py provides:
- TransformerTrainingConfig: Clean configuration management with validation
- UnivariateTransformerPipeline: Modular pipeline with state management  
- Better error handling and debugging capabilities
- Enhanced maintainability and reusability
"""
from src.main_train_univ_transformer_class import (
    TransformerUnivConfig,
    UnivariateTransformerPipeline,
)

from src.main_train_diagnostic_residual_transformer_class import(
    DiagnosticResidualTransformerConfig,
    DiagnosticResidualTransformerPipeline,
)

from src.main_train_seasonal_residual_transformer_class import(
    SeasonalResidualTransformerConfig,
    SeasonalResidualTransformerPipeline,
)

from src.univariate_transformer.utils_univ_transformer import load_mlflow_model_history 
from src.utils.experiments_utils import smart_read, safe_float, initialize_results_tracking, load_json_codes_list, cleanup_ram

# 2. Reemplazar la función original con nuestra función "inteligente"
pd.read_csv = smart_read


from src.config.base_transformer_config import BaseTransformerConfig


# MAIN TRANSFORMER MODEL WITH MLFLOW TRACKING
default_config = BaseTransformerConfig()
# Initialize MLflow
mlflow.set_tracking_uri("file:./mlruns")
experiment_name = f"Prova_SELFDiagnostics1_TRANSFORMERS_PREDAP"
mlflow.set_experiment(experiment_name)

print(f"🎯 MLflow tracking initialized")
print(f"   • Experiment: {experiment_name}")
print(f"   • Tracking URI: {mlflow.get_tracking_uri()}")
print(f"   • View results at: http://localhost:5000")

# Utility function for safe float conversion
def safe_float(value):
    """Convert value to float, handling numpy types and NaN values"""
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# MAIN TRANSFORMER MODEL 

LOOKBACK_LIST = default_config.LOOKBACK_LIST
FORECAST_LIST = default_config.FORECAST_LIST
CODES_LIST = ["demanda__TOTAL", "demanda__SERVEI_CODI__URG", "B34","J00", "I10", "M54","Ch01#subch01#A00-A09"]
CUTOFF_DATE = default_config.cutoff_date
ACTIVATIONS_LIST = default_config.ACTIVATIONS_LIST
covid_token = True
data_path = default_config.data_path

df = pd.read_parquet(default_config.data_path)

# Track overall experiment metrics
total_runs = len(CODES_LIST) * len(LOOKBACK_LIST) * len(FORECAST_LIST)
print(f"📊 Starting training for {total_runs} model configurations")

run_counter = 0

# Dictionary to track best results for each code
best_results_per_code = {}

# Initialize best results tracking for each code
for code in CODES_LIST:
    best_results_per_code[code] = {
        "best_mse": float('inf'),
        "best_config": None,
        "best_metrics": None,
        "best_run_info": None
    }

num_transformer_blocks = 2
head_size = 32
num_heads = 8
ff_dim = 512
mlp_units = [512,256] 

# Training hyperparameters
activation = "gelu"
covid_token =  True
dropout =  0.5
learning_rate = 0.00001


for CODE in CODES_LIST:
    for lb in LOOKBACK_LIST:
        for fh in FORECAST_LIST:
            run_counter += 1
            
            # Start MLflow run for this specific configuration
            run_name = f"Prova_SELFDiagnostics1_transformer_{CODE}_lb{lb}_fh{fh}_{datetime.now().strftime('%H%M%S')}"
            with mlflow.start_run(run_name=run_name) as run:
                print(f"\n🚀 [{run_counter}/{total_runs}] Starting MLflow run: {run_name}")
                print(f"   • Run ID: {run.info.run_id}")
                
                
                # Log hyperparameters
                mlflow.log_params({
                    "target_code": CODE,
                    "lookback": lb,
                    "forecast_horizon": fh,
                    "model_type": "transformer",
                    "run_number": run_counter,
                    "total_runs": total_runs,
                    "activation_function": activation,
                    "covid_token": covid_token,
                    "causal_masking": False,
                })
                
                # Log system information
                mlflow.log_params({
                    "tensorflow_version": tf.__version__,
                    "python_version": os.sys.version.split()[0],
                    "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0,
                    "backend": "Agg"  # matplotlib backend
                })
                lookback = lb
                forecast = fh
                code = CODE

                # ==================== PHASE 1: UNIVARIATE TRANSFORMER (CLASS-BASED) ====================
                univ_start_time = datetime.now()
                mlflow.log_param("phase_1_start_time", univ_start_time.isoformat())
                # Train univariate transformer and capture results
                batch_size = experiments_utils.compute_dynamic_batch_size(lookback, forecast)
                # Create configuration object
                transformer_config = TransformerUnivConfig(
                    lookback=lookback,
                    forecast=forecast,
                    code=code,
                    activation_function=activation,
                    covid_token= covid_token,   
                    cutoff_date=CUTOFF_DATE,
                    head_size = head_size, 
                    num_heads = num_heads, 
                    ff_dim = ff_dim, 
                    mlp_units = mlp_units,
                    evaluate_model = True,
                    positional_encoding = False,
                    data_path = data_path,
                    learning_rate = learning_rate,
                    batch_size = batch_size,
                )

                # Create and run pipeline
                pipeline = UnivariateTransformerPipeline(transformer_config)
                univ_outputs = pipeline.run_complete_pipeline()
                model = univ_outputs.model
                model_name = univ_outputs.model_name
                loss = univ_outputs.loss
                mae = univ_outputs.mae
                mse = univ_outputs.mse
                rmse = univ_outputs.rmse
                wape = univ_outputs.wape

                
                # Log the model and configuration details
                mlflow.keras.log_model(model, artifact_path="univariate_transformer")
                
                
                load_mlflow_model_history(model_name)


                univ_end_time = datetime.now()
                univ_duration = (univ_end_time - univ_start_time).total_seconds()
                
                mlflow.log_metrics({
                    "duration/phase_1_duration_seconds": univ_duration,
                    "duration/phase_1_duration_minutes": univ_duration / 60,

                })

                if loss is not None and mae is not None and mse is not None and rmse is not None and wape is not None:
                    mlflow.log_metrics({
                        "eval/univ_transformer_loss": loss,
                        "eval/univ_transformer_mae": mae,
                        "eval/univ_transformer_mse": mse,
                        "eval/univ_transformer_rmse": rmse,
                        "eval/univ_transformer_wape": wape
                    })
                print(f"\n\nRunning for Lookback: {lookback}, Forecast: {forecast}, Code: {code}\n")
                
                # RESIDUAL DIAGNOSTICS TRANSFORMER
                diag_start_time = datetime.now()
                mlflow.log_param("phase_2_start_time", diag_start_time.isoformat())
                
                # compact parameter grouping for readability
                diagnostic_parameters = DiagnosticResidualTransformerConfig(
                    lookback=lookback,
                    forecast=forecast,
                    code=code,
                    activation_function=activation,
                    covid_token=covid_token,
                    cutoff_date=CUTOFF_DATE,
                    predictions_train_corrected=None,
                    predictions_test_corrected=None,
                    head_size = head_size, 
                    num_heads = num_heads, 
                    ff_dim = ff_dim, 
                    mlp_units = mlp_units,
                    evaluate_model = True,
                    positional_encoding = False,
                    data_path = data_path,
                    learning_rate = 1e-4,
                    batch_size = batch_size,
                )

                pipeline = DiagnosticResidualTransformerPipeline(diagnostic_parameters)
                diagnostic_outputs = pipeline.run_complete_pipeline()
                predictions_train_corrected = diagnostic_outputs.predictions_train_corrected
                predictions_test_corrected = diagnostic_outputs.predictions_test_corrected
                residual_diagnostics_model = diagnostic_outputs.residual_diagnostics_model
                residual_diagnostics_model_name = diagnostic_outputs.residual_diagnostics_model_name
                corrected_diagnostics_mae = diagnostic_outputs.corrected_diagnostics_mae
                corrected_diagnostics_mse = diagnostic_outputs.corrected_diagnostics_mse
                corrected_diagnostics_rmse = diagnostic_outputs.corrected_diagnostics_rmse
                corrected_diagnostics_wape = diagnostic_outputs.corrected_diagnostics_wape

                
                                                                                                                                                                                                                                                                                                                
                mlflow.keras.log_model(residual_diagnostics_model, artifact_path="residual_diagnostics_model")
                load_mlflow_model_history(residual_diagnostics_model_name)
                
                diag_end_time = datetime.now()
                diag_duration = (diag_end_time - diag_start_time).total_seconds()
                
                mlflow.log_metrics({
                    "duration/phase_2_duration_seconds": diag_duration,
                    "duration/phase_2_duration_minutes": diag_duration / 60,
                    "eval/residual_diagnostics_model_mae": corrected_diagnostics_mae,
                    "eval/residual_diagnostics_model_mse": corrected_diagnostics_mse,    
                    "eval/residual_diagnostics_model_rmse": corrected_diagnostics_rmse,
                    "eval/residual_diagnostics_model_wape": corrected_diagnostics_wape,
                    })
                # RESIDUAL SEASONAL TRANSFORMER
                seasonal_start_time = datetime.now()
                mlflow.log_param("phase_3_start_time", seasonal_start_time.isoformat())

                # compact parameter grouping for readability
                seasonal_params = SeasonalResidualTransformerConfig(
                    lookback=lookback,
                    forecast=forecast,
                    code=code,
                    activation_function=activation,
                    covid_token=covid_token,
                    cutoff_date=CUTOFF_DATE,
                    predictions_train_corrected=predictions_train_corrected,
                    predictions_test_corrected=predictions_test_corrected,
                    batch_size = batch_size,
                )
                pipeline = SeasonalResidualTransformerPipeline(seasonal_params)
                seasonal_outputs = pipeline.run_complete_pipeline()
                predictions_train_corrected = seasonal_outputs.predictions_train_corrected
                predictions_test_corrected = seasonal_outputs.predictions_test_corrected
                residual_seasonal_model = seasonal_outputs.residual_diagnostics_model
                residual_seasonal_model_name = seasonal_outputs.residual_diagnostics_model_name
                corrected_seasonal_mae = seasonal_outputs.corrected_diagnostics_mae
                corrected_seasonal_mse = seasonal_outputs.corrected_diagnostics_mse 
                corrected_seasonal_rmse = seasonal_outputs.corrected_diagnostics_rmse
                corrected_seasonal_wape = seasonal_outputs.corrected_diagnostics_wape

                

                mlflow.keras.log_model(residual_seasonal_model, artifact_path="residual_seasonal_model")
                load_mlflow_model_history(residual_seasonal_model_name)
                
                seasonal_end_time = datetime.now()
                seasonal_duration = (seasonal_end_time - seasonal_start_time).total_seconds()
                total_duration = (seasonal_end_time - univ_start_time).total_seconds()
                
                mlflow.log_metrics({
                    "duration/phase_3_duration_seconds": seasonal_duration,
                    "duration/phase_3_duration_minutes": seasonal_duration / 60,
                    "total_training_duration_seconds": total_duration,
                    "total_training_duration_minutes": total_duration / 60,
                    "eval/residual_seasonal_model_mae": corrected_seasonal_mae,
                    "eval/residual_seasonal_model_mse": corrected_seasonal_mse,
                    "eval/residual_seasonal_model_rmse": corrected_seasonal_rmse,
                    "eval/residual_seasonal_model_wape": corrected_seasonal_wape,
                })
    
                
                # Use the final seasonal MSE as the comparison metric (best overall performance)
                current_mse = corrected_seasonal_mse
                
                print(f"✅ [{run_counter}/{total_runs}] Completed run for {CODE} - lb:{lb} fh:{fh}")


# ==================== FINAL SUMMARY AND CONSOLIDATED RESULTS ====================

print(f"\n🎉 GRID SEARCH COMPLETED!")
print(f"📊 Total runs: {total_runs}")


