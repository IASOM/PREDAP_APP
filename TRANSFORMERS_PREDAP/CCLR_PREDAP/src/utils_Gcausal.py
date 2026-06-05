# utils_Gcausal.py
# ------------------------------------------------------
# Utility functions for time series Granger causality analysis
# Author: Guillem Hernández Guillamet
# Version: 1.0
# Date: 04/06/2025
# Description:
#   This module contains helper functions for smoothing time series,
#   testing stationarity (KPSS), generating lag plots, selecting optimal VAR lags,
#   and computing Granger causality matrices between multiple variables.
# ------------------------------------------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from statsmodels.tsa.stattools import kpss, grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR

from gcausal import (
    smoother, stationate, min_max_scale,
    lag_plots,
    kpss_test,
    splitter, select_p,
    granger_causation_matrix,
    fit_var_model,
    recommend_lag_order,
    select_causal_features
)



if __name__ == "__main__":
    # RAW DATA --------------------------------------------------------------
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the project root, then into data folder
    data_path = os.path.join(script_dir, '..', 'data', 'synthetic_timeseries.csv')
    df = pd.read_csv(data_path, index_col=0)
    df.index = pd.date_range(start="2010-01-01", periods=len(df), freq="D")
    df = df.clip(lower=0)

    # SMOOTHING DATA --------------------------------------------------------
    WINDOW_SIZE = 14
    '''smoothed = smoother(df, WINDOW_SIZE)'''
    # SCALED DATA ----------------------------------------------------------
    smoothed_scaled = min_max_scale(df)

    smoothed = smoothed_scaled

    # Define our target series 
    target = ["timeseries_350"]

    # Define our predictor series
    predictors = ['timeseries_405','timeseries_363','timeseries_975','timeseries_120','timeseries_2','timeseries_541','timeseries_775',
                'timeseries_181','timeseries_692','timeseries_443','timeseries_763','timeseries_31','timeseries_324']
    
    
    ### 0.1. Stationarity (check)
    variables = [target+predictors]
    variables = [item for sublist in variables for item in sublist]
    lag_plots(df[variables]) # lag plots to visualize trends

    # KPSS test
    print(kpss_test(df[variables]))

    indexes = kpss_test(df[variables]).T[kpss_test(df[variables]).T['p-value'] < 0.05].index.tolist()
    print(indexes)

    # first order stationarity
    stationate_df = stationate(df[variables],indexes)
    lag_plots(stationate_df[variables[0:len(variables)]])

    print(kpss_test(stationate_df))
    indexes = kpss_test(stationate_df).T[kpss_test(stationate_df).T['p-value'] < 0.05].index.tolist()
    print(indexes)

    # second order stationarity
    stationate_df = stationate(stationate_df,indexes)
    lag_plots(stationate_df[variables[0:len(variables)]])

    print(kpss_test(stationate_df))


    ### 0.2. G-Causality test
    train_df, test_df = splitter(stationate_df)
    results_df, optimal_lags = select_p(train_df)
    opt_lag = recommend_lag_order(optimal_lags=optimal_lags)# Optimal lag based on criteria from previous step


    var_result = fit_var_model(train_df, opt_lag)

    print(granger_causation_matrix(train_df, train_df.columns, p=1))
    print(granger_causation_matrix(train_df, train_df.columns, p=7))
    print(granger_causation_matrix(train_df, train_df.columns, p=30))

    gmatrix = granger_causation_matrix(train_df, train_df.columns, p=opt_lag)
    causal_features = select_causal_features(gmatrix, target_variable=target[0], significance_level=0.05)

    print(gmatrix)
    print("The selected causal features are:", causal_features)