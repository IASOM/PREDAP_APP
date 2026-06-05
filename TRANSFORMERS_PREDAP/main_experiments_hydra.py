import pandas as pd
import os
from src.utils.experiments_utils import (
    smart_read, safe_float, initialize_results_tracking, 
    load_json_codes_list, cleanup_ram, memory_cleanup
    )

# 2. Reemplazar la función original con nuestra función "inteligente"
pd.read_csv = smart_read

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend (no GUI)
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.tensorflow
import mlflow.keras
from datetime import datetime

import tempfile
import json
import pickle
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.global_hydra import GlobalHydra
from tensorflow.keras import backend as K
import gc
from src.utils import experiments_utils


#from src.univariate_transformer import default_config
from src.utils import data_preparation



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

from src.config.base_transformer_config import BaseTransformerConfig

default_config = BaseTransformerConfig()
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# Register custom resolver for loading JSON codes list
OmegaConf.register_new_resolver("load_json_codes_list", load_json_codes_list)
config_name = "config_production.yaml"  

@hydra.main(version_base=None, config_path="conf", config_name=config_name)
def main_experiment(cfg: DictConfig) -> None:
    """Main experiment function decorated with Hydra for parameter sweeping."""
    
    # Initialize MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    experiment_name = f"{cfg.mlflow.experiment_name}_{datetime.now().strftime('%Y%m%d')}"
    mlflow.set_experiment(experiment_name)
    
    print(f"🎯 MLflow tracking initialized")
    print(f"   • Experiment: {experiment_name}")
    print(f"   • Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"   • View results at: http://localhost:5000")
    
    # Initialize results tracking (only once per sweep)
    codes_list = load_json_codes_list(cfg.data.codes_path)
    
    # Extract parameters from Hydra config
    CODE = cfg.model.target_code
    lookback = cfg.model.lookback
    forecast = cfg.model.forecast
    head_size = cfg.model.head_size
    num_heads = cfg.model.num_heads
    ff_dim = cfg.model.ff_dim
    mlp_units = cfg.model.mlp_units
    activation_function = cfg.model.activation
    covid_token = cfg.model.covid_token
    cutoff_date = cfg.training.cutoff_date
    positional_encoding = cfg.training.positional_encoding
    data_path = cfg.data.data_path
    evaluate_model=cfg.training.evaluate_model
    num_transformer_blocks = cfg.model.num_transformer_blocks
    dropout = cfg.model.dropout
    learning_rate = cfg.model.learning_rate
    scaler = default_config.scaler

    
    # Start MLflow run for this specific configuration
    run_name = f"1.0_Production_TRANSFORMER_{CODE}_lb{lookback}_fh{forecast}_{datetime.now().strftime('%H%M%S')}" #TRANSFORMER3
    with mlflow.start_run(run_name=run_name) as run:
        print(f"\n🚀 Starting MLflow run: {run_name}")
        print(f"   • Run ID: {run.info.run_id}")
        
        # Log all hyperparameters from config
        mlflow.log_params({
            "target_code": CODE,
            "lookback": lookback,
            "forecast_horizon": forecast,
            "model_type": "transformer",
            "dataset": data_path,
            "activation_function": activation_function,
            "covid_token": covid_token,
            "cutoff_date": cutoff_date,
            "head_size": head_size,
            "num_heads": num_heads, 
            "ff_dim": ff_dim,
            "mlp_units": mlp_units,
            "num_transformer_blocks": num_transformer_blocks,
            "dropout": dropout,
            "learning_rate": learning_rate ,
            "positional_encoding": positional_encoding,
            "causal_masking": False,
            "scaler": scaler,
        })
        
        # Log system information
        mlflow.log_params({
            "tensorflow_version": tf.__version__,
            "python_version": os.sys.version.split()[0],
            "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0,
            "backend": cfg.system.backend,
            "hydra_config_name": cfg._target_ if hasattr(cfg, '_target_') else "config"
        })

        # ==================== PHASE 1: UNIVARIATE TRANSFORMER ====================
        univ_start_time = datetime.now()
        mlflow.log_param("phase_1_univariate_transformer_start_time", univ_start_time.isoformat())
        batch_size = experiments_utils.compute_dynamic_batch_size(lookback, forecast)
        
        # Train univariate transformer with parameters from config
        univariate_parameters = TransformerUnivConfig(  
            lookback=lookback,
            forecast=forecast,
            code=CODE,
            activation_function=activation_function,
            covid_token=covid_token,   
            cutoff_date=cutoff_date,
            head_size=head_size, 
            num_heads=num_heads, 
            ff_dim=ff_dim, 
            mlp_units=mlp_units,
            evaluate_model=evaluate_model,
            positional_encoding=positional_encoding,
            num_transformer_blocks=num_transformer_blocks,
            dropout=dropout,
            learning_rate=learning_rate,
            data_path=data_path,
            batch_size = batch_size,     
        )

        pipeline = UnivariateTransformerPipeline(univariate_parameters)
        univ_outputs = pipeline.run_complete_pipeline()
        model = univ_outputs.model
        model_name = univ_outputs.model_name
        loss = float(univ_outputs.loss)
        mae = float(univ_outputs.mae)
        mse = float(univ_outputs.mse)
        rmse = float(univ_outputs.rmse)
        wape = float(univ_outputs.wape)

        mlflow.keras.log_model(model, artifact_path="univariate_model")
        del model  # Clear model from memory before loading history to free up GPU memory
        
        load_mlflow_model_history(model_name, model_type="univariate_transformer")

        univ_end_time = datetime.now()
        univ_duration = (univ_end_time - univ_start_time).total_seconds()
        univ_duration = float(univ_duration)  # Ensure it's a standard float for MLflow logging


        

        mlflow.log_metrics({
            "duration/phase_1_univariate_transformer_duration_seconds": univ_duration,
            "duration/phase_1_univariate_transformer_duration_minutes": univ_duration / 60,
        })

        if loss is not None and mae is not None and mse is not None:
            mlflow.log_metrics({
                "eval/univ_transformer_loss": loss,
                "eval/univ_transformer_mae": mae,
                "eval/univ_transformer_mse": mse,
                "eval/univ_transformer_rmse": rmse,
                "eval/univ_transformer_wape": wape
            })
        
        print(f"\n\nRunning for Lookback: {lookback}, Forecast: {forecast}, Code: {CODE}\n")

        #Clear the GPU memory and possible memory garbage
        del univ_outputs
        del pipeline
        memory_cleanup()
        plt.close('all')
        
        # ==================== PHASE 2: RESIDUAL DIAGNOSTICS TRANSFORMER ====================
        diag_start_time = datetime.now()
        mlflow.log_param("phase_2_residual_diagnostics_transformer_start_time", diag_start_time.isoformat())
        
        # Train residual diagnostics transformer with parameters from config
        diagnostic_parameters = DiagnosticResidualTransformerConfig(
            lookback=lookback,
            forecast=forecast,
            code=CODE,
            activation_function=activation_function,
            covid_token=covid_token,
            cutoff_date=cutoff_date,
            predictions_train_corrected=None,
            predictions_test_corrected=None,
            head_size=head_size,
            num_heads=num_heads,
            ff_dim=ff_dim,
            mlp_units=mlp_units,
            dropout=dropout,
            learning_rate=learning_rate,
            data_path=data_path,
            batch_size = batch_size,
            num_transformer_blocks=num_transformer_blocks,
        )

        pipeline = DiagnosticResidualTransformerPipeline(diagnostic_parameters)
        diagnostic_outputs = pipeline.run_complete_pipeline()
        predictions_train_corrected = np.array(diagnostic_outputs.predictions_train_corrected, copy = True)
        predictions_test_corrected = np.array(diagnostic_outputs.predictions_test_corrected, copy = True)
        residual_diagnostics_model = diagnostic_outputs.residual_diagnostics_model
        residual_diagnostics_model_name = diagnostic_outputs.residual_diagnostics_model_name
        corrected_diagnostics_mae = float(diagnostic_outputs.corrected_diagnostics_mae)
        corrected_diagnostics_mse = float(diagnostic_outputs.corrected_diagnostics_mse)
        corrected_diagnostics_rmse = float(diagnostic_outputs.corrected_diagnostics_rmse)
        corrected_diagnostics_wape = float(diagnostic_outputs.corrected_diagnostics_wape)



        mlflow.keras.log_model(residual_diagnostics_model, artifact_path="residual_diagnostics_model")
        del residual_diagnostics_model  # Clear model from memory before loading history to free up GPU memory
        load_mlflow_model_history(residual_diagnostics_model_name, model_type="residual_diagnostics_transformer")
        
        diag_end_time = datetime.now()
        diag_duration = (diag_end_time - diag_start_time).total_seconds()
        
        mlflow.log_metrics({
            "duration/phase_2_residual_diagnostics_transformer_duration_seconds": float(diag_duration),
            "duration/phase_2_residual_diagnostics_transformer_duration_minutes": float(diag_duration / 60),
            "eval/residual_diagnostics_model_mae": float(corrected_diagnostics_mae),
            "eval/residual_diagnostics_model_mse": float(corrected_diagnostics_mse),
            "eval/residual_diagnostics_model_rmse": float(corrected_diagnostics_rmse),
            "eval/residual_diagnostics_model_wape": float(corrected_diagnostics_wape),
        })
        
        #Clear the GPU memory and possible memory garbage
        del diagnostic_outputs
        del pipeline
        memory_cleanup()
        plt.close('all')
        
        # ==================== PHASE 3: RESIDUAL SEASONAL TRANSFORMER ====================
        seasonal_start_time = datetime.now()
        mlflow.log_param("phase_3_residual_seasonal_transformer_start_time", seasonal_start_time.isoformat())

        # Train residual seasonal transformer with parameters from config
        seasonal_params = SeasonalResidualTransformerConfig(
            lookback=lookback,
            forecast=forecast,
            code=CODE,
            activation_function=activation_function,
            covid_token=covid_token,
            cutoff_date=cutoff_date,
            predictions_train_corrected= predictions_train_corrected,
            predictions_test_corrected= predictions_test_corrected,
            head_size=head_size,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_transformer_blocks=num_transformer_blocks,
            mlp_units = mlp_units,
            dropout=dropout, #Only applyied to the transformer
            learning_rate=learning_rate,#Not applyied yet
            data_path=data_path,
            batch_size = batch_size,
            )

        pipeline = SeasonalResidualTransformerPipeline(seasonal_params)
        seasonal_outputs = pipeline.run_complete_pipeline()
        predictions_train_corrected = np.array(seasonal_outputs.predictions_train_corrected, copy = True)
        predictions_test_corrected = np.array(seasonal_outputs.predictions_test_corrected, copy = True)
        residual_seasonal_model = seasonal_outputs.residual_diagnostics_model
        residual_seasonal_model_name = seasonal_outputs.residual_diagnostics_model_name
        corrected_seasonal_mae = float(seasonal_outputs.corrected_diagnostics_mae)
        corrected_seasonal_mse = float(seasonal_outputs.corrected_diagnostics_mse)
        corrected_seasonal_rmse = float(seasonal_outputs.corrected_diagnostics_rmse)
        corrected_seasonal_wape = float(seasonal_outputs.corrected_diagnostics_wape)

        mlflow.keras.log_model(residual_seasonal_model, artifact_path="residual_seasonal_model")
        load_mlflow_model_history(residual_seasonal_model_name, model_type="residual_seasonal_transformer")
        
        seasonal_end_time = datetime.now()
        seasonal_duration = (seasonal_end_time - seasonal_start_time).total_seconds()
        total_duration = (seasonal_end_time - univ_start_time).total_seconds()
        
        mlflow.log_metrics({
            "duration/phase_3_residual_seasonal_transformer_duration_seconds": float(seasonal_duration),
            "duration/phase_3_residual_seasonal_transformer_duration_minutes": float(seasonal_duration / 60),
            "total_training_duration_seconds": float(total_duration),
            "total_training_duration_minutes": float(total_duration / 60),
            "eval/residual_seasonal_model_mae": float(corrected_seasonal_mae),
            "eval/residual_seasonal_model_mse": float(corrected_seasonal_mse),
            "eval/residual_seasonal_model_rmse": float(corrected_seasonal_rmse),
            "eval/residual_seasonal_model_wape": float(corrected_seasonal_wape),
        })
        
        # Use the final seasonal MSE as the comparison metric
        current_mse = corrected_seasonal_mse
        
        # Log final metrics for this run
        mlflow.log_metrics({
            "final/seasonal_mse": float(current_mse),
            "final/seasonal_mae": float(corrected_seasonal_mae),
            "final/seasonal_rmse": float(corrected_seasonal_rmse),
            "final/seasonal_wape": float(corrected_seasonal_wape),
        })
        
        print(f"✅ Completed run for {CODE} - lb:{lookback} fh:{forecast}")
        print(f"   • Seasonal MSE: {current_mse:.6f}")
        print(f"   • Configuration: head_size={head_size}, num_heads={num_heads}, ff_dim={ff_dim}, mlp_units={mlp_units}")
        #Clear the GPU memory and possible memory garbage
        del seasonal_outputs
        del pipeline
        del residual_seasonal_model, univariate_parameters, seasonal_params, predictions_train_corrected, predictions_test_corrected
        memory_cleanup()
        plt.close('all')
        
if __name__ == "__main__":
    main_experiment()
    cleanup_ram()