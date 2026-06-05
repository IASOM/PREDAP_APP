"""
Residual Multivariate Transformers Package
==========================================

This package contains modules for implementing residual correction using multivariate
transformer models. It follows a modular structure for better code organization and
maintainability.

Modules:
    - config_residual_transformer: Configuration parameters and settings
    - model_architecture_residual_transformer: Model architectures and learning rate schedulers  
    - training_evaluation_residual_transformer: Training and evaluation utilities
    - utils_residual_transformer: Data processing and utility functions
    - visualization_residual_transformer: Plotting and visualization functions
    - main_training_residual_transformer: Main training and evaluation pipeline
"""

from .model_architecture_residual_transformer import (
    transformer_encoder,
    hybrid_lstm_transformer_model,
)
from .training_evaluation_residual_transformer import (
    train_given_model_and_data,
    load_trained_model,
    setup_gpu_memory,
    create_model_directories,
    compare_model_performance,
    save_performance_results,
)
from .utils_residual_transformer import (
    prepare_base_model_data,
    load_base_model_transformer,
    prepare_residual_data,
    split_train_test,
    filter_diagnostics_covariates,
)
from .visualization_residual_transformer import (
    plot_residuals_analysis,
    plot_stepwise_errors_comparison,
    plot_predictions_with_pandemic_waves,
    plot_errors_over_time_with_waves,
    evaluate_error_significance_pandemic_waves,
    create_pandemic_waves_df,
    plot_training_history,
    plot_model_comparison,
    
)

__all__ = [
    'transformer_encoder',
    'hybrid_lstm_transformer_model',
    'train_given_model_and_data',
    'load_trained_model',
    'prepare_base_model_data',
    'plot_residuals_analysis',
    'plot_stepwise_errors_comparison',
    'plot_predictions_with_pandemic_waves',
    'plot_errors_over_time_with_waves',
    'evaluate_error_significance_pandemic_waves',
    'create_pandemic_waves_df',
    'plot_training_history',
    'plot_model_comparison',
]

__version__ = "1.0.0"
__author__ = "TRANSFORMERS_PREDAP Team"