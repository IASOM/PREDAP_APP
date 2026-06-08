import os
from datetime import datetime
import pandas as pd
from src.config.base_transformer_config import BaseTransformerConfig as default_config
import json
import matplotlib.pyplot as plt
import gc
import ctypes
from tensorflow.keras import backend as K
import tensorflow as tf
from typing import List

import re

from src.training.training_residual_transformer import load_trained_model


_original_read_csv = pd.read_csv
_in_smart_read = False

def cleanup_ram():
    plt.close('all')
    K.clear_session()
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

def smart_read(file_path, **kwargs):
    """
    Función que sustituye a pd.read_csv.
    Detecta automáticamente si la extensión es .parquet o .csv
    y llama a la función de lectura apropiada.
    """
    if str(file_path).lower().endswith('.parquet'):
        print(f"-> INFO: Leyendo {file_path} como PARQUET.")
        # Aquí puedes añadir parámetros específicos para Parquet si los necesitas
        return pd.read_parquet(file_path, **kwargs)
    else:
        # Llama a la función original pd.read_csv para CSVs y otros
        print(f"-> INFO: Leyendo {file_path} como CSV (o formato predeterminado).")
        return _original_read_csv(file_path, **kwargs)

# Utility function for safe float conversion
def safe_float(value):
    """Convert value to float, handling numpy types and NaN values"""
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
    
def initialize_results_tracking(results_dir: str, codes_list) -> dict:
    """Initialize result tracking directories and structures."""
    
    os.makedirs(results_dir, exist_ok=True)
    print(f"📁 Results will be saved in: {results_dir}/")
    
    # Get all possible target codes from default config for initialization
    best_results_per_code = {}
    for code in codes_list:
        best_results_per_code[code] = {
            "best_mse": float('inf'),
            "best_config": None,
            "best_metrics": None,
            "best_run_info": None
        }

    return best_results_per_code

def load_json_codes_list(json_path: str) -> str:
    """Load a list from JSON and return as comma-separated string for Hydra sweep."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    #codes_list = data[key]
    # Return comma-separated string for Hydra sweep parameters
    return ','.join(data)

def get_codes_list(input_directory: str) -> str:
    """
    Reads the input data file, extracts unique codes, and returns them
    individually wrapped in quotes as a comma-separated string for Hydra sweeps.
    """
    if input_directory.endswith('.csv'):
        df = pd.read_csv(input_directory, nrows=0)
        codes_list = df.columns.tolist()
    elif input_directory.endswith('.parquet'):
        import pyarrow.parquet as pq
        schema = pq.read_schema(input_directory)
        codes_list = schema.names
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Parquet file.")

    cleaned_codes = []
    for code in codes_list:
        clean_str = str(code).strip()
        
        # Filter out invalid columns
        if clean_str and clean_str != 'timestamp' and not clean_str.startswith('__index'):
            # CRUCIAL FIX: Wrap the code in escaped quotes so Hydra treats 
            # "BARCELONA CIUTAT" as a single literal string item.
            cleaned_codes.append(f'"{clean_str}"')
            
    # This will return: "DEMAND_DEMANDA_TOTAL","DEMAND_... BARCELONA CIUTAT","..."
    return ",".join(cleaned_codes)


def compute_dynamic_batch_size(lookback, forecast):
    """
    Computes an appropriate batch size based on lookback and forecast parameters.

    Parameters:
    - lookback (int): Number of past timesteps used for input sequences.
    - forecast (int): Number of timesteps predicted.

    Returns:
    - int: Computed batch size.
    """
    gpus = tf.config.list_physical_devices('GPU')

    if lookback <= 14 and forecast <= 14:
        batch_size = 4096
    if lookback <= 30 and forecast <= 30:
        batch_size = 1024
    elif (30 <= lookback <= 60) and forecast <= 60:
        batch_size = 256
    elif 60 <= lookback <= 128 and forecast<= 128:
        batch_size = 128
    elif 128 < lookback <= 365 and forecast <=365:
        batch_size = 128
    else:
        batch_size = 92
    

    if len(gpus) == 0:
        batch_size = 32
        
    return batch_size


def memory_cleanup():
    """
    Clean up GPU memory between training phases.
    
    NOTE: Do NOT use numba cuda.device.reset() — it destroys the CUDA context
    and makes TensorFlow unable to use the GPU for the rest of the process.
    """
    # Clear Keras session (releases model graphs and cached tensors)
    K.clear_session()
    
    # Reset the default graph (TF1 compat, still useful for freeing resources)
    tf.compat.v1.reset_default_graph()
    
    # Force Python garbage collection
    gc.collect()
    
    # Close all matplotlib figures to free memory
    plt.close('all')
    
    # Free unused C memory back to the OS
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    
    # Reset GPU memory stats (does NOT destroy the CUDA context)
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.reset_memory_stats(gpu)
        except Exception:
            pass


def extract_model_params(model_name):
    """
    Extract lookback and forecast parameters from model filename.
    
    Expected format: {code}_example_transformer_{forecast}fh_{ff_dim}ff_{lookback}lb_{lr}initlr.keras
    
    Args:
        model_name (str): Model filename
        
    Returns:
        tuple: (lookback, forecast) or (None, None) if not found
    """
    # Pattern to match the model name format
    pattern = r'(\d+)fh_\d+ff_(\d+)lb_'
    
    match = re.search(pattern, model_name)
    if match:
        forecast = int(match.group(1))  # First group is forecast
        lookback = int(match.group(2))  # Second group is lookback
        return lookback, forecast
    else:
        print(f"Could not extract parameters from: {model_name}")
        return None, None





