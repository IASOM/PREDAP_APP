# Deep Learning Package for Time Series Prediction
# ==================================================
# Author: Guillem Hernández Guillamet  
# Version: 1.0
# Date: 04/06/2025
# Description: Modular deep learning utilities for time series forecasting

# Import all functions from submodules
from .preprocessing_dl import add_temprality, split_sequence
from .models_dl import (
    create_model_gru, 
    create_model_lstm, 
    create_model_bilstm, 
    create_model_enc_dec, 
    create_model_enc_dec_cnn,
    create_model_vector_output, 
    create_model_multi_head_cnn_lstm
)
from .training_dl import fit_model, auto_grid_search
from .prediction_dl import prediction, inverse_transform
from .evaluation_dl import evaluate_forecast, get_results
from .visualization_dl import plot_train_test, plt_model


# Define what gets imported with "from dl import *"
__all__ = [
    # Preprocessing
    'add_temprality',
    'split_sequence',
    
    # Models
    'create_model_gru',
    'create_model_lstm', 
    'create_model_bilstm',
    'create_model_enc_dec',
    'create_model_enc_dec_cnn',
    'create_model_vector_output',
    'create_model_multi_head_cnn_lstm',
    
    # Training
    'fit_model',
    
    # Prediction
    'prediction',
    'inverse_transform',
    
    # Evaluation
    'evaluate_forecast',
    'get_results',
    
    # Visualization
    'plot_train_test',
    'plt_model',
    
    # Optimization
    'auto_grid_search'
]