# utils_LMLR.py
# ------------------------------------------------------
# Utility functions for lagged multiple regression models
# Author: Guillem Hernández Guillamet
# Version: 1.0
# Date: 04/06/2025
# ------------------------------------------------------



from IPython.display import display
import os
import pandas as pd
import numpy as np
import scipy.stats
import math
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

# Import functions from the modular LMLR package
from lmlr import (
    # Preprocessing functions
    smoother, min_max_scale, infection_index_df,
    
    # Visualization functions
    plot_example, ploter, plot_models, plot_metrics,
    
    # Correlation and VIF functions
    get_top_correlations_blog, compute_vif, filter_VIF,
    
    # Modeling functions
    models_training, select_best_models, select_best_absolute_model,
    
    # Metrics functions
    metrics_calculation, evaluation_metrics_MAPE, 
    evaluation_metrics_RMSE, evaluation_metrics_Ftest
)


if __name__ == "__main__":
    #HYPERPARAMETERS
    # DIAG
    CODE = 'timeseries_350'
    # SMOOTHING
    WINDOW_SIZE = 14
    # CORR
    CORRELATION_THRESHOLD = 0.90 # Above this value, variables are considered strongly correlated
    # VIF
    ITERATIONS_MAX = 400
    ITERATIONS = 0
    VIF_THRESHOLD = 20.0

    #MODELS 
    MAX_ITERS_MODEL = 60
    MAX_LAG = 30 


    # RAW DATA --------------------------------------------------------------
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the project root, then into data folder
    data_path = os.path.join(script_dir, '..', 'data', 'synthetic_timeseries.csv')
    df = smart_read(data_path, index_col=0)
    df.index = pd.date_range(start="2010-01-01", periods=len(df), freq="D")
    df = df.clip(lower=0)

    # SMOOTHING DATA --------------------------------------------------------
    '''smoothed =smoother(df, WINDOW_SIZE)
    plot_example(smoothed, "SMOOTHED (example 10 ts)")'''

    # SCALED DATA ----------------------------------------------------------
    # Min-Max Scaling
    smoothed = min_max_scale(df) # normalize df
    plot_example(smoothed, "SMOOTHED + SCALED DATA (example 10 ts)")


    # FILTER VARIABLES ---------------------------------------------------
    # Make a copy of the smoothed dataframe
    dataframe = smoothed.copy()

    # Compute top correlations above a threshold
    df_correlations = get_top_correlations_blog(dataframe, threshold=CORRELATION_THRESHOLD)
    print(df_correlations)

    # Extract correlated variable names from both levels of the MultiIndex
    correlated_vars = list(set(df_correlations.index.get_level_values(0).tolist() + 
                            df_correlations.index.get_level_values(1).tolist()))

    # Compute variables not involved in any strong correlation
    non_correlated_vars = list(set(dataframe.columns) - set(correlated_vars))

    # variables que són subjecte de ser eliminades
    redundant_vars = correlated_vars


    # VIF ---------------------------------------------------
    # compute VIF de les variables no correlacionades
    vif = compute_vif(redundant_vars, dataframe).sort_values('VIF', ascending=False)
    print(vif)

    # filtrar el vif
    redundant_vars = filter_VIF(vif, dataframe, ITERATIONS_MAX, VIF_THRESHOLD)



    # MODELS CONSTRUCTION ------------------------------------------------------
    # sort absolute correlations between variables and cov19
    smot_corr = df.corrwith(smoothed[CODE]).sort_values(ascending=False, key=abs)


    # A. initial model (on-time prediction) ----------------
    df_init = models_training(dataframe, CODE, smot_corr, MAX_ITERS_MODEL)


    # B. lagged models (1 to MAX_LAG days ahead) ----------------
    
    #BEST = df_init[df_init["BEST_MODEL"] == "YES"]
    #BEST["LAG"] = 0 
    BEST = df_init

    for i in range(1,MAX_LAG+1):
        dat_lag = dataframe.copy()
        # add lag
        dat_lag[CODE] = dat_lag[CODE].shift(-i)
        dat_lag = dat_lag.dropna(subset=[CODE])

        # OBTENIR EL MILLOR MODEL
        df_init = models_training(dat_lag, CODE, smot_corr, MAX_ITERS_MODEL, plt_selected_models=True)
        best = df_init[df_init["BEST_MODEL"] == "YES"]
        best["LAG"] = i
        BEST = pd.concat([BEST,best]).reset_index(drop=True)

    print(BEST)

    # Save best features into an Excel file 
    BEST.to_excel("BEST_features_NOSMOOTH_timeseries350.xlsx", index=False, engine='openpyxl')

    #EVALUATION METRICS ANALYISIS ------------------------------------------------------
    evaluation_metrics_MAPE(BEST)
    evaluation_metrics_RMSE(BEST)
    evaluation_metrics_Ftest(BEST)


    # PREDICTION LAGGED FIRST MODEL ------------------------------------------------------
    # Asses how features selected for lag=0 model behave with different lags
    Y_PREDICTIONS_test = []  # training results
    Y_PREDICTIONS_train = [] # testing results
    no_lag_predictors = True

    for lagging in range(0,MAX_LAG):
        
        # predictors del millor model per a lag == lagging (0,30) o inicial
            # varien en funció del model
        if no_lag_predictors == True:
            predictors = BEST.loc[0,"predictors"].split(",")
        else:
            predictors = BEST.loc[lagging,"predictors"].split(",")
        
        # copia de la bbbdd
        df = dataframe.copy()
        split_index = round(len(df)*0.8) 
        df = df[[CODE]+predictors]
        
        # Realitzem una copia lagged de la bbdd
        if lagging > 0:
            df[CODE] = df[CODE].shift(-lagging)
            df = df.dropna(subset=[CODE])
        print("TRAINING ROWS post lagging % s" % str(df.shape))

        
        #train_test_split
        split_date = df.index[split_index-lagging]
        df_train = df.loc[df.index <= split_date].copy()
        df_test = df.loc[df.index > split_date].copy()
        y_train = df_train[CODE].values
        y_test = df_test[CODE].values
        X_train = df_train
        X_train = X_train[predictors]
        X_test = df_test
        X_test = X_test[predictors]

        # model training
        ols_model = sm.OLS(y_train,X_train)
        ols_results = ols_model.fit()

        #populate vars
        y_pred_train = ols_results.predict(X_train)
        y_pred_test = ols_results.predict(X_test)
        Y_PREDICTIONS_test.append(y_pred_test)
        Y_PREDICTIONS_train.append(y_pred_train)

    # plot all models
    fig = plt.figure()
    fig.suptitle(CODE + ' prediction')
    lines = [] 

    for i in range(len(Y_PREDICTIONS_test)):#-1):
        lines += plt.plot(df_test.index, Y_PREDICTIONS_test[i], label='%s day lagging' % str(i+1))
    lines += plt.plot(df_test.index, y_test, 'go-', label='Actual %s diagnoses' % CODE)
    labels = [l.get_label() for l in lines]
    plt.legend(lines, labels) 
    plt.xlabel('Date')
    plt.ylabel('n diagnoses')
    plt.show()
