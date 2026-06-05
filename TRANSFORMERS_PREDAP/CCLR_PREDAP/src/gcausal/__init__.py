# gcausal package
# ------------------------------------------------------
# Granger Causality Analysis utilities package
# Author: Guillem Hernández Guillamet
# Version: 1.0
# Date: 04/06/2025
# ------------------------------------------------------

"""
GCAUSAL (Granger Causality Analysis) Package

This package provides utilities for time series Granger causality analysis,
including data preprocessing, stationarity testing, visualization,
and causal relationship detection.

Modules:
    - preprocessing_gcausal: Data smoothing and stationarity transformation
    - visualization_gcausal: Lag plots and diagnostic visualizations
    - stationarity_gcausal: KPSS tests and stationarity analysis
    - causality_gcausal: Granger causality testing and matrix generation
    - model_selection_gcausal: VAR model selection and lag optimization
"""

# Import main functions for easy access
from .preprocessing_gcausal import smoother, stationate, min_max_scale
from .visualization_gcausal import lag_plots, plot_var_metrics, plot_granger_matrix
from .stationarity_gcausal import kpss_test, identify_nonstationary_series, stationarity_summary
from .causality_gcausal import granger_causation_matrix,granger_causation_matrix_parallel, interpret_causality_matrix, filter_causality_matrix
from .model_selection_gcausal import splitter, select_p, fit_var_model, recommend_lag_order, select_causal_features

__all__ = [
    # Preprocessing
    'smoother', 'stationate', 'min_max_scale',
    
    # Visualization
    'lag_plots', 'plot_var_metrics', 'plot_granger_matrix',
    
    # Stationarity Testing
    'kpss_test', 'identify_nonstationary_series', 'stationarity_summary',
    
    # Causality Analysis
    'granger_causation_matrix','granger_causation_matrix_parallel', 'interpret_causality_matrix', 'filter_causality_matrix',
    
    # Model Selection
    'splitter', 'select_p', 'fit_var_model', 'recommend_lag_order', 'select_causal_features'
]

__version__ = "1.0.0"
__author__ = "Guillem Hernández Guillamet"