"""
Save Final Predictions Module
=============================
Functions to generate and save predictions from trained univariate transformer models.
"""

import os
import sys
import glob
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import FunctionTransformer

# Add the src directory to path for module imports - MUST be before src imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from data_utils import data_preparation
from src.univariate_transformer.utils_univ_transformer import extract_model_params
from model_architechture.model_architecture_univ_transformer import (
    RevIN, PositionalEncoding
)
from src.config.base_transformer_config import BaseTransformerConfig

# Default configuration 
default_config = BaseTransformerConfig()

# Custom objects for model loading
CUSTOM_OBJECTS = {
    "RevIN": RevIN,
    "PositionalEncoding": PositionalEncoding,
    #"LearnableQuery": LearnableQuery,
}


def load_historic_data(data_path: str = "data/EXAMPLE/historic_df.xlsx") -> pd.DataFrame:
    """
    Load the historic data from an Excel file.
    
    Parameters:
    -----------
    data_path : str
        Path to the historic Excel file
        
    Returns:
    --------
    pd.DataFrame
        Historic data with timestamp column
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Historic data file not found at: {data_path}")
    
    df = pd.read_excel(data_path)
    print(f"Loaded historic data with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df


def extract_code_from_model_name(model_name: str) -> Optional[str]:
    """
    Extract the code from a model filename.
    The code is everything before '_base_transformer'.
    
    Parameters:
    -----------
    model_name : str
        Model filename (e.g., 'J00_base_transformer_lb90_fh7.keras')
        
    Returns:
    --------
    str or None
        The extracted code (e.g., 'J00'), or None if pattern not found
    """
    if "_base_transformer" in model_name:
        code = model_name.split("_base_transformer")[0]
        return code
    return None


def discover_model_folders(base_path: str = "models") -> Dict[str, str]:
    """
    Discover all code-specific model folders.
    
    Parameters:
    -----------
    base_path : str
        Base path where model folders are located
        
    Returns:
    --------
    Dict[str, str]
        Dictionary mapping code names to folder paths
    """
    code_folders = {}
    
    if not os.path.exists(base_path):
        print(f"Warning: Base model path {base_path} does not exist")
        return code_folders
    
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            # Extract code from folder name (e.g., "J00_cat_model" -> "J00")
            code = folder.replace("_cat_model", "")
            code_folders[code] = folder_path
            print(f"Found model folder for code {code}: {folder_path}")
    
    return code_folders


'''def discover_models_in_folder(folder_path: str) -> Dict[int, str]:
    """
    Discover all .keras models in a folder and organize by forecast horizon.
    
    Parameters:
    -----------
    folder_path : str
        Path to the model folder
        
    Returns:
    --------
    Dict[int, str]
        Dictionary mapping forecast horizon to model file path
    """
    models_by_forecast = {}
    
    keras_files = glob.glob(os.path.join(folder_path, "*.keras"))
    
    for model_path in keras_files:
        model_name = os.path.basename(model_path)
        lookback, forecast = extract_model_params(model_name)
        
        if forecast is not None:
            models_by_forecast[forecast] = model_path
            print(f"  Found model: {model_name} (forecast={forecast}, lookback={lookback})")
    
    return models_by_forecast'''

def discover_models_in_folder(folder_path: str) -> Dict[int, str]:
    """
    Discover all .keras models in a folder (recursively) and organize by forecast horizon.
    Only includes models with 'base_transformer' in the name.
    
    Parameters:
    -----------
    folder_path : str
        Path to the model folder (searches recursively)
        
    Returns:
    --------
    Dict[int, str]
        Dictionary mapping forecast horizon to model file path
    """
    models_by_forecast = {}
    
    # Search recursively for .keras files
    keras_files = glob.glob(os.path.join(folder_path, "**/*.keras"), recursive=True)
    # Also check direct path
    keras_files += glob.glob(os.path.join(folder_path, "*.keras"))
    
    for model_path in keras_files:
        model_name = os.path.basename(model_path)
        
        # Only process base_transformer models
        if "base_transformer" not in model_name:
            continue
        
        # Extract code from model name (everything before _base_transformer)
        code = extract_code_from_model_name(model_name)
        
        lookback, forecast = extract_model_params(model_name)
        
        if forecast is not None:
            models_by_forecast[forecast] = model_path
            print(f"  Found model: {model_name} (code={code}, forecast={forecast}, lookback={lookback})")
    
    return models_by_forecast


def discover_all_models(base_path: str = "models") -> Dict[str, Dict[int, str]]:
    """
    Discover all models and organize by code and forecast horizon.
    Extracts code from model filename (everything before '_base_transformer').
    
    Parameters:
    -----------
    base_path : str
        Base path where model files are located (can be flat or nested)
        
    Returns:
    --------
    Dict[str, Dict[int, str]]
        Dictionary mapping code -> {forecast_horizon -> model_path}
    """
    models_by_code = {}
    
    if not os.path.exists(base_path):
        print(f"Warning: Base model path {base_path} does not exist")
        return models_by_code
    
    # Search for .keras files recursively
    keras_files = glob.glob(os.path.join(base_path, "**/*.keras"), recursive=True)
    keras_files += glob.glob(os.path.join(base_path, "*.keras"))  # Also check base path directly
    
    for model_path in keras_files:
        model_name = os.path.basename(model_path)
        
        # Only process base_transformer models
        if "base_transformer" not in model_name:
            continue
        
        # Extract code from model name (everything before _base_transformer)
        code = extract_code_from_model_name(model_name)
        if code is None:
            continue
            
        lookback, forecast = extract_model_params(model_name)
        
        if forecast is not None:
            if code not in models_by_code:
                models_by_code[code] = {}
            models_by_code[code][forecast] = model_path
            print(f"  Found model: {model_name} (code={code}, forecast={forecast}, lookback={lookback})")
    
    return models_by_code


def load_model_safe(
    model_path: str,
    timeout_seconds: int = 120,
    head_size_list: List[int] = [32, 64],
    num_heads_list: List[int] = [4, 8,16],
    ff_dim_list: List[int] = [512],
    num_transformer_blocks_list: List[int] = [2, 4, 6]
) -> Optional[tf.keras.Model]:
    """
    Load a Keras model with custom objects, handling potential errors.
    For models with problematic custom layers, loads weights into a fresh model.
    
    Tries different parameter combinations if layer count mismatch occurs.
    
    Parameters:
    -----------
    model_path : str
        Path to the model file
    timeout_seconds : int
        Timeout for standard model loading
    head_size_list : List[int]
        List of head sizes to try for weights-only loading
    num_heads_list : List[int]
        List of number of attention heads to try
    ff_dim_list : List[int]
        List of feed-forward dimensions to try
    num_transformer_blocks_list : List[int]
        List of number of transformer blocks to try
    """
    import zipfile
    import threading
    import h5py
    
    # 1. Check file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return None
    
    # 2. Detect file format
    file_format = "unknown"
    try:
        with open(model_path, 'rb') as f:
            header = f.read(8)
            if header[:4] == b'PK\x03\x04':
                file_format = "keras_v3_zip"
            elif header[:4] == b'\x89HDF':
                file_format = "hdf5"
        print(f"  Detected format: {file_format} for {os.path.basename(model_path)}")
    except Exception as e:
        print(f"Warning: Could not read file header: {e}")
    
    # 3. Clear previous session
    tf.keras.backend.clear_session()
    tf.config.run_functions_eagerly(True)
    
    # 4. First try weights-only load with parameter search (works for both formats)
    try:
        print("  Attempting weights-only load with parameter search...")
        model = load_model_weights_only(
            model_path,
            head_size_list=head_size_list,
            num_heads_list=num_heads_list,
            ff_dim_list=ff_dim_list,
            num_transformer_blocks_list=num_transformer_blocks_list
        )
        if model is not None:
            print(f"  Successfully loaded model via weights-only method")
            tf.config.run_functions_eagerly(False)
            return model
    except Exception as e:
        print(f"  Weights-only load failed: {e}")
    
    # 5. Fall back to standard load with subprocess isolation
    print("  Falling back to standard load...")
    model = load_model_in_subprocess(model_path, timeout_seconds)
    
    tf.config.run_functions_eagerly(False)
    return model


def load_model_weights_only(
    model_path: str,
    head_size_list: List[int] = [32, 64, 128],
    num_heads_list: List[int] = [4, 8],
    ff_dim_list: List[int] = [256, 512],
    num_transformer_blocks_list: List[int] = [2, 4, 6]
) -> Optional[tf.keras.Model]:
    """
    Load model by extracting config from HDF5 and loading weights into fresh model.
    This bypasses the RevIN graph scope issue.
    
    Tries different combinations of model parameters if layer count mismatch occurs.
    
    Parameters:
    -----------
    model_path : str
        Path to the .keras or .h5 model file
    head_size_list : List[int]
        List of head sizes to try
    num_heads_list : List[int]
        List of number of attention heads to try
    ff_dim_list : List[int]
        List of feed-forward dimensions to try
    num_transformer_blocks_list : List[int]
        List of number of transformer blocks to try
        
    Returns:
    --------
    tf.keras.Model or None
        Loaded model, or None if loading fails
    """
    import h5py
    import json
    from itertools import product
    from model_architechture.model_architecture_univ_transformer import build_model
    
    # Get lookback and forecast from filename
    model_name = os.path.basename(model_path)
    lookback, forecast = extract_model_params(model_name)
    
    if lookback is None or forecast is None:
        print(f"  Could not extract lookback/forecast from filename")
        return None
    
    print(f"  Building fresh model with lookback={lookback}, forecast={forecast}")
    
    # Generate all parameter combinations to try
    param_combinations = list(product(
        head_size_list,
        num_heads_list, 
        ff_dim_list,
        num_transformer_blocks_list
    ))
    
    print(f"  Will try up to {len(param_combinations)} parameter combinations...")
    
    last_error = None
    for i, (head_size, num_heads, ff_dim, num_blocks) in enumerate(param_combinations):
        try:
            # Clear session before each attempt to avoid state issues
            tf.keras.backend.clear_session()
            
            # Build model with these parameters
            model = build_model(
                input_shape=(lookback, 14),
                head_size=head_size,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_transformer_blocks=num_blocks,
                mlp_units=[512, 256],
                activation_function="gelu",
                dropout=0.1,
                mlp_dropout=0.1,
                n_pred=forecast,
                pos_encoding=True
            )
            
            # Try to load weights
            model.load_weights(model_path)
            print(f"  ✓ Success with params: head_size={head_size}, num_heads={num_heads}, ff_dim={ff_dim}, num_blocks={num_blocks}")
            return model
            
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            # Check if this is a layer/shape mismatch error - these are expected, keep trying
            if any(keyword in error_str for keyword in ["layer count mismatch", "shape", "mismatch", "incompatible", "expected"]):
                if i < 10:  # Print first 10 failures
                    print(f"    Combo {i+1}/{len(param_combinations)} failed: hs={head_size}, nh={num_heads}, ff={ff_dim}, blocks={num_blocks}")
                elif i == 10:
                    print(f"    ... (suppressing further mismatch messages)")
                continue
            else:
                # Unexpected error - log but continue trying
                print(f"    Combo {i+1} unexpected error: {type(e).__name__}: {str(e)[:100]}")
                continue
    
    print(f"  ✗ All {len(param_combinations)} parameter combinations failed")
    if last_error:
        print(f"    Last error was: {last_error}")
    return None


def load_model_in_subprocess(model_path: str, timeout_seconds: int) -> Optional[tf.keras.Model]:
    """
    Try to load model in a way that can be killed if it hangs.
    Returns None if loading fails or times out.
    """
    import threading
    import multiprocessing as mp
    
    # Use a simple thread with timeout - if it hangs, we skip this model
    model = None
    error_msg = None
    load_complete = threading.Event()
    
    def load_thread():
        nonlocal model, error_msg
        try:
            model = tf.keras.models.load_model(
                model_path,
                compile=False,
                custom_objects=CUSTOM_OBJECTS
            )
        except Exception as e:
            error_msg = str(e)
        finally:
            load_complete.set()
    
    thread = threading.Thread(target=load_thread, daemon=True)
    thread.start()
    
    # Wait with timeout
    success = load_complete.wait(timeout=timeout_seconds)
    
    if not success:
        print(f"  Model loading timed out after {timeout_seconds}s")
        print(f"  This model has the RevIN graph scope bug and cannot be loaded.")
        print(f"  You need to retrain this model with the fixed RevIN layer.")
        return None
    
    if model is None and error_msg:
        print(f"  Error loading model: {error_msg}")
    
    return model

def prepare_input_data(
    data_path: str,
    code: str,
    lookback: int,
    forecast: int,
    scaler=None,
    cutoff_date: str = "2008-01-01",
    max_date: str = "2027-09-30",
    covid_token: bool = True,
    eliminate_covid_data: bool = False,
    covid_dates: list = None,
    diagnostic_covariates_path: str = None,
    split_ratio: float = 0.8
) -> np.ndarray:
    """
    Prepare input data for prediction using the same pipeline as training.
    Includes temporal encodings (day_of_week, month, season, holiday, etc.)
    
    Parameters:
    -----------
    data_path : str
        Path to the data file (CSV or parquet)
    code : str
        Target code column name
    lookback : int
        Number of past days to use as input
    forecast : int
        Forecast horizon (needed to load diagnostic covariates)
    scaler : sklearn transformer, optional
        Scaler for normalization (use identity if None)
    cutoff_date : str
        Start date for data
    max_date : str
        End date for data
    covid_token : bool
        Whether to include COVID token feature
    eliminate_covid_data : bool
        Whether to remove COVID period data
    covid_dates : list
        List of COVID period date ranges
    diagnostic_covariates_path : str, optional
        Path to diagnostic covariates file (without code suffix)
        
    Returns:
    --------
    np.ndarray
        Input array of shape (1, lookback, features) - last sequence for prediction
    """
    code = code.replace("#", ":")
    
    # Load diagnostic covariates if path provided
    relevant_feature_cols = None
    if diagnostic_covariates_path is not None:
        try:
            full_path = diagnostic_covariates_path + code + ".xlsx"
            if os.path.exists(full_path):
                diagnostic_df = pd.read_excel(full_path, engine='openpyxl')
                matching_rows = diagnostic_df[diagnostic_df['LAG'] == forecast]
                if len(matching_rows) > 0:
                    relevant_feature_cols = matching_rows['predictors'].iloc[0].split(',')
                    print(f"    Loaded {len(relevant_feature_cols)} diagnostic covariates")
        except Exception as e:
            print(f"    Warning: Could not load diagnostic covariates: {e}")
    
    # Use the same data preparation function as training
    # This ensures temporal features are included
    X, Y = data_preparation.prepare_data(
        data_path,
        code,
        lookback,
        forecast,
        covid_token=covid_token,
        cutoff_date=cutoff_date,
        max_date=max_date,
        train=False,  # Use test split to get most recent data
        debug=False,
        univariate=True,
        scaler=scaler,
        eliminate_covid_data=eliminate_covid_data,
        covid_dates=covid_dates,
        relevant_feature_cols=relevant_feature_cols,
        split_ratio=split_ratio
    )
    
    # Return only the last sequence (most recent data for prediction)
    if len(X) > 0:
        X_input = X[-1:, :, :]  # Shape: (1, lookback, features)
        print(f"    Input shape: {X_input.shape}")
        return X_input
    else:
        raise ValueError(f"No data available for code {code} with given parameters")


def generate_predictions_single_model(
    model: tf.keras.Model,
    X_input: np.ndarray,
    forecast: int
) -> np.ndarray:
    """
    Generate predictions using a single model.
    
    Parameters:
    -----------
    model : tf.keras.Model
        Trained model
    X_input : np.ndarray
        Input data of shape (1, lookback, features)
    forecast : int
        Expected forecast horizon
        
    Returns:
    --------
    np.ndarray
        Predictions of shape (forecast,)
    """
    predictions = model.predict(X_input, verbose=0)
    predictions = np.maximum(predictions.flatten()[:forecast], 0)  # Ensure non-negative
    return predictions


def generate_predictions_for_code(
    data_path: str,
    code: str,
    models_by_forecast: Dict[int, str],
    scaler=None,
    cutoff_date: str = "2008-01-01",
    max_date: str = "2027-09-30",
    covid_token: bool = False,
    eliminate_covid_data: bool = False,
    covid_dates: list = None,
    diagnostic_covariates_path: str = None,
    split_ratio: float = 0.8
) -> Dict[int, np.ndarray]:
    """
    Generate predictions for all forecast horizons for a single code.
    
    Parameters:
    -----------
    data_path : str
        Path to the data file (CSV or parquet)
    code : str
        Target code
    models_by_forecast : Dict[int, str]
        Dictionary mapping forecast horizon to model path
    scaler : sklearn transformer, optional
        Scaler for normalization
    cutoff_date : str
        Start date for data filtering
    max_date : str
        End date for data filtering
    covid_token : bool
        Whether to include COVID token feature
    eliminate_covid_data : bool
        Whether to remove COVID period data
    covid_dates : list
        List of COVID period date ranges
    diagnostic_covariates_path : str, optional
        Path to diagnostic covariates file (without code suffix)
        
    Returns:
    --------
    Dict[int, np.ndarray]
        Dictionary mapping forecast horizon to predictions array
    """
    predictions_by_horizon = {}
    
    for forecast, model_path in sorted(models_by_forecast.items()):
        print(f"  Processing forecast horizon: {forecast}")
        
        # Get lookback from model name
        lookback, _ = extract_model_params(os.path.basename(model_path))
        if lookback is None:
            print(f"    Warning: Could not extract lookback from {model_path}")
            continue
        
        # Load model
        model = load_model_safe(model_path)
        if model is None:
            continue
        
        # Prepare input data using full data preparation pipeline
        try:
            X_input = prepare_input_data(
                data_path=data_path,
                code=code,
                lookback=lookback,
                forecast=forecast,
                scaler=scaler,
                cutoff_date=cutoff_date, 
                max_date=max_date,
                covid_token=covid_token,
                eliminate_covid_data=eliminate_covid_data,
                covid_dates=covid_dates,
                diagnostic_covariates_path=diagnostic_covariates_path,
                split_ratio=split_ratio
            )
        except Exception as e:
            print(f"    Warning: Could not prepare input data: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Generate predictions
        try:
            preds = generate_predictions_single_model(model, X_input, forecast)
            predictions_by_horizon[forecast] = preds
            print(f"    Generated {len(preds)} predictions")
        except Exception as e:
            print(f"    Warning: Prediction failed: {e}")
        
        # Clear session to free memory
        tf.keras.backend.clear_session()
    
    return predictions_by_horizon


def save_predictions_by_horizon(
    predictions_by_horizon: Dict[int, np.ndarray],
    code: str,
    output_dir: str = "predictions_output"
) -> str:
    """
    Save predictions to Excel with each forecast horizon as a column.
    
    Parameters:
    -----------
    predictions_by_horizon : Dict[int, np.ndarray]
        Dictionary mapping forecast horizon to predictions
    code : str
        Target code (used for filename)
    output_dir : str
        Output directory for Excel files
        
    Returns:
    --------
    str
        Path to saved Excel file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find max forecast length
    max_len = max(len(preds) for preds in predictions_by_horizon.values()) if predictions_by_horizon else 0
    
    # Create DataFrame with day index
    df_out = pd.DataFrame({"Day": range(1, max_len + 1)})
    
    # Add each horizon as a column
    for horizon in sorted(predictions_by_horizon.keys()):
        preds = predictions_by_horizon[horizon]
        # Pad with NaN if needed
        padded = np.full(max_len, np.nan)
        padded[:len(preds)] = preds
        df_out[f"Horizon_{horizon}d"] = padded
    
    # Save to Excel
    output_path = os.path.join(output_dir, f"{code}_predictions_by_horizon.xlsx")
    df_out.to_excel(output_path, index=False)
    print(f"Saved predictions to: {output_path}")
    
    return output_path


def create_combined_forecast(
    predictions_by_horizon: Dict[int, np.ndarray]
) -> np.ndarray:
    """
    Create a combined forecast where each segment uses the appropriate model.
    
    For example:
    - Days 1-7: use 7-day model
    - Days 8-14: use 14-day model (taking days 8-14 of its predictions)
    - Days 15-30: use 30-day model (taking days 15-30)
    - etc.
    
    Parameters:
    -----------
    predictions_by_horizon : Dict[int, np.ndarray]
        Dictionary mapping forecast horizon to predictions array
        
    Returns:
    --------
    np.ndarray
        Combined forecast array
    """
    if not predictions_by_horizon:
        return np.array([])
    
    sorted_horizons = sorted(predictions_by_horizon.keys())
    max_horizon = max(sorted_horizons)
    
    # Initialize combined array
    combined = np.full(max_horizon, np.nan)
    
    # Fill in segments
    prev_end = 0
    for horizon in sorted_horizons:
        preds = predictions_by_horizon[horizon]
        
        # Determine the segment this model covers
        # Each model fills from prev_end to min(horizon, len(preds))
        start_idx = prev_end
        end_idx = min(horizon, len(preds))
        
        if start_idx < end_idx and start_idx < len(preds):
            combined[start_idx:end_idx] = preds[start_idx:end_idx]
        
        prev_end = end_idx
    
    return combined


def save_combined_predictions(
    combined_predictions: Dict[str, np.ndarray],
    output_dir: str = "predictions_output"
) -> str:
    """
    Save combined predictions for all codes to a single Excel file.
    
    Parameters:
    -----------
    combined_predictions : Dict[str, np.ndarray]
        Dictionary mapping code to combined predictions array
    output_dir : str
        Output directory
        
    Returns:
    --------
    str
        Path to saved Excel file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find max length
    max_len = max(len(preds) for preds in combined_predictions.values()) if combined_predictions else 0
    
    # Create DataFrame
    df_out = pd.DataFrame({"Day": range(1, max_len + 1)})
    
    for code in sorted(combined_predictions.keys()):
        preds = combined_predictions[code]
        padded = np.full(max_len, np.nan)
        padded[:len(preds)] = preds
        df_out[code] = padded
    
    # Save to Excel
    output_path = os.path.join(output_dir, "combined_predictions_all_codes.xlsx")
    df_out.to_excel(output_path, index=False)
    print(f"Saved combined predictions to: {output_path}")
    
    return output_path


def run_full_prediction_pipeline(
    code: str,
    data_path: str = "../data/diagnostics_CAT_aggregated.parquet",
    models_base_path: str = "J00_CAT_model",
    output_dir: str = "best_hyperparameters_results",
    scaler=None,
    cutoff_date: str = "2008-01-01",
    max_date: str = "2026-09-30",
    covid_token: bool = True,
    eliminate_covid_data: bool = False,
    covid_dates: list = None,
    diagnostic_covariates_path: str = None,
    split_ratio: float = 0.8
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """
    Run the complete prediction pipeline for a specific code across all forecast horizons.
    
    Parameters:
    -----------
    code : str
        The target code to generate predictions for (e.g., 'J00', 'T14')
    data_path : str
        Path to data file (CSV or parquet) - used for preparing input with temporal features
    models_base_path : str
        Base path where model files are located
    output_dir : str
        Directory to save output Excel files
    scaler : sklearn transformer, optional
        Scaler for data normalization (default: identity)
    cutoff_date : str
        Start date for data filtering
    max_date : str
        End date for data filtering
    covid_token : bool
        Whether to include COVID token feature
    eliminate_covid_data : bool
        Whether to remove COVID period data
    covid_dates : list
        List of COVID period date ranges
    diagnostic_covariates_path : str, optional
        Path to diagnostic covariates file (without code suffix)
        
    Returns:
    --------
    Tuple[Dict, np.ndarray]
        - predictions_by_horizon: Dict[horizon, predictions]
        - combined_predictions: combined_predictions_array
    """
    print("=" * 60)
    print(f"STARTING PREDICTION PIPELINE FOR CODE: {code}")
    print("=" * 60)
    
    # Use identity scaler if none provided
    if scaler is None:
        scaler = FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)
    
    # Step 1: Verify data path exists
    print(f"\n[Step 1] Verifying data path: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    print(f"  Data file found: {data_path}")
    
    # Step 2: Discover models for this code
    print(f"\n[Step 2] Discovering models for code: {code}")
    models_by_forecast = discover_models_in_folder(models_base_path)
    
    if not models_by_forecast:
        print(f"No models found for {code}. Please check the models path.")
        return {}, np.array([])
    
    print(f"  Found {len(models_by_forecast)} models for horizons: {sorted(models_by_forecast.keys())}")
    
    # Step 3: Generate predictions for all horizons
    print(f"\n[Step 3] Processing code: {code}")
    print("-" * 40)
    
    predictions_by_horizon = generate_predictions_for_code(
        data_path=data_path,
        code=code,
        models_by_forecast=models_by_forecast,
        scaler=scaler,
        cutoff_date=cutoff_date,
        max_date=max_date,
        covid_token=covid_token,
        eliminate_covid_data=eliminate_covid_data,
        covid_dates=covid_dates,
        diagnostic_covariates_path=diagnostic_covariates_path,
        split_ratio=split_ratio
    )
    
    combined = np.array([])
    
    if predictions_by_horizon:
        # Save individual code predictions
        save_predictions_by_horizon(predictions_by_horizon, code, output_dir)
        
        # Create combined forecast
        combined = create_combined_forecast(predictions_by_horizon)
        
        # Save combined predictions to separate Excel file
        combined_output_path = os.path.join(output_dir, f"{code}_combined_forecast.xlsx")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create DataFrame with day index and predictions
        df_combined = pd.DataFrame({
            "Day": range(1, len(combined) + 1),
            "Prediction": combined
        })
        df_combined.to_excel(combined_output_path, index=False)
        print(f"\nSaved combined forecast to: {combined_output_path}")
        
        # Print combined predictions
        print(f"\n{'='*60}")
        print(f"COMBINED FORECAST FOR {code} ({len(combined)} days)")
        print(f"{'='*60}")
        print(f"{'Day':<8} {'Prediction':>15}")
        print(f"{'-'*8} {'-'*15}")
        for day, pred in enumerate(combined, 1):
            if not np.isnan(pred):
                print(f"{day:<8} {pred:>15.2f}")
            else:
                print(f"{day:<8} {'NaN':>15}")
        print(f"{'='*60}")
    
    print("\n" + "=" * 60)
    print("PREDICTION PIPELINE COMPLETE")
    print("=" * 60)
    
    return predictions_by_horizon, combined


def plot_combined_forecast(
    excel_path: str,
    code: str = None,
    save_path: str = None,
    figsize: Tuple[int, int] = (14, 6)
) -> None:
    """
    Plot the combined forecast from an Excel file.
    
    Parameters:
    -----------
    excel_path : str
        Path to the combined forecast Excel file (e.g., 'J00_combined_forecast.xlsx')
    code : str, optional
        Code name for the title. If None, extracts from filename.
    save_path : str, optional
        Path to save the plot. If None, displays interactively.
    figsize : Tuple[int, int]
        Figure size (width, height)
    """
    import matplotlib.pyplot as plt
    
    # Load the Excel file
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    
    df = pd.read_excel(excel_path)
    
    # Extract code from filename if not provided
    if code is None:
        filename = os.path.basename(excel_path)
        code = filename.replace("_combined_forecast.xlsx", "")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot predictions
    days = df["Day"].values
    predictions = df["Prediction"].values
    
    # Plot line
    ax.plot(days, predictions, 'b-', linewidth=1.5, label='Combined Forecast')
    
    # Fill area under curve
    ax.fill_between(days, predictions, alpha=0.3)
    
    # Mark different horizon segments if visible
    # Add markers at key points
    valid_mask = ~np.isnan(predictions)
    if valid_mask.any():
        ax.scatter(days[valid_mask][::7], predictions[valid_mask][::7], 
                   c='red', s=20, zorder=5, label='Weekly markers')
    
    # Formatting
    ax.set_xlabel('Forecast Day', fontsize=12)
    ax.set_ylabel('Predicted Value', fontsize=12)
    ax.set_title(f'Combined Forecast for {code}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Set x-axis limits
    ax.set_xlim(1, len(days))
    
    # Add statistics annotation
    valid_preds = predictions[~np.isnan(predictions)]
    if len(valid_preds) > 0:
        stats_text = f"Min: {valid_preds.min():.1f}\nMax: {valid_preds.max():.1f}\nMean: {valid_preds.mean():.1f}"
        ax.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Example usage for a single code
    code = "J00"  # Specify the target code
    
    predictions_by_horizon, combined_preds = run_full_prediction_pipeline(
        code=code,
        data_path='../data/FINAL_DB/full_CAT1.parquet',  # Main data file with all codes
        models_base_path="../data/EXAMPLE/J00_CAT_model",
        output_dir="best_hyperparameters_results",
        scaler=None,  # Uses identity scaler
        cutoff_date="2008-01-01",
        max_date="2026-09-30",
        covid_token=True,
        eliminate_covid_data=False,
        covid_dates=None,
        diagnostic_covariates_path="../data/best_features/BEST_features_NOSMOOTH_"  # Without code suffix
    )
    
    # Print summary
    print(f"\nSummary for {code}:")
    print(f"  Horizons: {sorted(predictions_by_horizon.keys())} days")
    print(f"  Combined forecast length: {len(combined_preds)} days")
    
    # Plot the combined forecast
    excel_path = f"best_hyperparameters_results/{code}_combined_forecast.xlsx"
    if os.path.exists(excel_path):
        plot_combined_forecast(
            excel_path=excel_path,
            code=code,
            save_path=f"best_hyperparameters_results/{code}_combined_forecast_plot.png"
        )
