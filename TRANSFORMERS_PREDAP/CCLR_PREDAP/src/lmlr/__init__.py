# lmlr package
# ------------------------------------------------------
# Lagged Multiple Linear Regression utilities package
# Author: Guillem Hernández Guillamet
# Version: 1.0
# Date: 04/06/2025
# ------------------------------------------------------

"""
LMLR (Lagged Multiple Linear Regression) Package

This package provides utilities for time series analysis using lagged multiple
linear regression models, including data preprocessing, feature selection,
model training, and evaluation.

Modules:
    - preprocessing: Data smoothing, scaling, and transformation utilities
    - visualization: Plotting functions for data exploration and results
    - correlation: Correlation analysis and multicollinearity detection
    - modeling: Model training, selection, and prediction functions  
    - metrics: Evaluation metrics and performance analysis
"""

# Import main functions for easy access
from .preprocessing_lmlr import smoother, min_max_scale, infection_index_df
from .visualization_lmlr import plot_example, ploter, plot_models, plot_metrics
from .correlation_lmlr import get_top_correlations_blog, compute_vif, filter_VIF, compute_vif_matrix, compute_vif_matrix_gpu
from .modeling_lmlr import models_training, select_best_models, select_best_absolute_model
from .metrics_lmlr import metrics_calculation, evaluation_metrics_MAPE, evaluation_metrics_RMSE, evaluation_metrics_Ftest

__all__ = [
    # Preprocessing
    'smoother', 'min_max_scale', 'infection_index_df',
    
    # Visualization  
    'plot_example', 'ploter', 'plot_models', 'plot_metrics',
    
    # Correlation
    'get_top_correlations_blog', 'compute_vif', 'filter_VIF', 'compute_vif_matrix', 'compute_vif_matrix_gpu'
    
    # Modeling
    'models_training', 'select_best_models', 'select_best_absolute_model',
    
    # Metrics
    'metrics_calculation', 'evaluation_metrics_MAPE', 
    'evaluation_metrics_RMSE', 'evaluation_metrics_Ftest'
]

__version__ = "1.0.0"
__author__ = "Guillem Hernández Guillamet"