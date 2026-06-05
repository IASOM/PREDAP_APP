"""
Univariate Transformer Package
=============================
A comprehensive package for univariate time series forecasting using transformer models.

This package contains:
- model_architecture_univ_transformer: Transformer model architecture and components
- training_evaluation_univ_transformer: Training and evaluation functions
- visualization_univ_transformer: Plotting and visualization utilities
- utils_univ_transformer: Helper functions and utilities
- config_univ_transformer: Configuration and parameter management
- main_training_univ_transformer: Main training and evaluation pipeline

Example usage:
    from config_univ_transformer import default_config
    from model_architecture_univ_transformer import build_model
    from training_evaluation_univ_transformer import train_given_model_and_data
    
    # Load configuration
    config = default_config
    
    # Build model
    model = build_model(
        (config.LOOKBACK, 1),
        head_size=config.HEAD_SIZE,
        num_heads=config.NUM_HEADS,
        ff_dim=config.FF_DIM,
        num_transformer_blocks=config.NUM_TRANSFORMER_BLOCKS,
        mlp_units=[config.MLP_UNITS],
        mlp_dropout=config.MLP_DROPOUT,
        dropout=config.DROPOUT,
        n_pred=config.FORECAST
    )
"""

# Import key components for easy access
from .model_architecture_univ_transformer import build_model, CustomCosineDecay
from .training_evaluation_univ_transformer import train_given_model_and_data, evaluate_model_sliding_window
from .visualization_univ_transformer import plt_model, plot_predictions_with_waves, plot_example
from .utils_univ_transformer import extract_model_params, load_and_evaluate_models, setup_gpu_memory, create_model_directories, create_pandemic_waves_df, load_and_preprocess_data
from .evaluation_univ_transformer import evaluate_univ_transformer
__version__ = "1.0.0"
__author__ = "TRANSFORMERS_PREDAP Team"

__all__ = [

    # Model Architecture
    'build_model',
    'CustomCosineDecay',
    
    # Training & Evaluation
    'train_given_model_and_data',
    'evaluate_model_sliding_window',
    
    # Visualization
    'plt_model',
    'plot_predictions_with_waves',
    'plot_example',
    
    # Utilities
    'extract_model_params',
    'load_and_evaluate_models',
    'setup_gpu_memory',
    'create_model_directories',
    'create_pandemic_waves_df',
    'load_and_preprocess_data',

    # Evaluation
    'evaluate_univ_transformer'
    
]