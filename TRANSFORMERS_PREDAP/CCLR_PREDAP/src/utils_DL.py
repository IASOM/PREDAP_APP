# utils_DL.py
# ------------------------------------------------------
# Utility functions for deep learning-based time series prediction
# Author: Guillem Hernández Guillamet
# Version: 2.0 - Modular Version
# Date: 04/06/2025
# Description:
#   This module imports functions from the modular dl package for
#   deep learning-based time series prediction with backward compatibility.
# ------------------------------------------------------

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# Import from modular dl package
from dl import (
    # Preprocessing
    add_temprality, split_sequence,
    
    # Models
    create_model_gru, create_model_lstm, create_model_bilstm,
    create_model_enc_dec, create_model_enc_dec_cnn,
    create_model_vector_output, create_model_multi_head_cnn_lstm,
    
    # Training
    fit_model,
    
    # Prediction
    prediction, inverse_transform,
    
    # Evaluation
    evaluate_forecast, get_results,
    
    # Visualization
    plot_train_test, plt_model,
    
    # Optimization
    auto_grid_search
)

# Import from LMLR package for data processing
from lmlr import smoother, min_max_scale

# Global variables that will be used by the models
FORECAST_RANGE = 7
n_features = None
col_idx = None


if __name__ == "__main__":
    # RAW DATA --------------------------------------------------------------
    '''# Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the project root, then into data folder
    data_path = os.path.join(script_dir, '..', 'data', 'synthetic_timeseries.csv')
    df = smart_read(data_path, index_col=0)
    df.index = pd.date_range(start="2010-01-01", periods=len(df), freq="D")
    df = df.clip(lower=0)'''

    BEST_FEATURES_PATH = 'BEST_features_NOSMOOTH.xlsx'

    data_path = 'J:/longitudinalitat_DIAGNOSTICS_GROUPED.csv'
    df = smart_read(data_path, index_col=0)
    df.index = pd.date_range(start="2010-01-01",periods=len(df), freq="D")
    df = df.clip(lower=0)

    
    # SMOOTHING DATA --------------------------------------------------------
    WINDOW_SIZE = 14
    '''smoothed = smoother(df, WINDOW_SIZE)
    # SCALED DATA ----------------------------------------------------------
    smoothed_scaled = min_max_scale(smoothed)

    smoothed = smoothed_scaled'''
    
    smoothed = df

    # Objective and predictors ...................................
    objective = "J00"
    cc_predictors = pd.read_excel(BEST_FEATURES_PATH, engine='openpyxl')
    cc_predictors = (list(cc_predictors['predictors'])[14]).split(",")
    # df definition ..............................................
    cc_predictors.append(objective)
    variables_predictors = cc_predictors.copy()
    subdf = smoothed[cc_predictors]

    # Add temporality (weekday and month) ........................
    TEMPORALITY = False
    if TEMPORALITY:
        subdf, cc_predictors = add_temprality(subdf, cc_predictors)

    # put objective to last column ..............................
    cc_predictors = [col for col in subdf.columns if col != objective]
    cc_predictors.append(objective)
    subdf = subdf[cc_predictors]
    print(subdf) 

    # train-test split ...........................................
    TRAIN_PERCENTAGE = 0.8
    train_size = int(len(subdf)*TRAIN_PERCENTAGE)
    train_dataset, test_dataset = subdf.iloc[5:train_size],subdf.iloc[train_size:]

    # Plot train and test data ...................................
    plot_train_test(train_dataset, test_dataset, objective, show_plt = True)

    #scale ......................................................
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_dataset)
    scaled_test = scaler.transform(test_dataset)

    # get idx of objective variable
    col_idx = subdf.columns.get_loc(objective)

    # DEEP LEARNING PARAMETERS ..........................
    LOOK_BACK =  30 #731 (2 anys)     # quants dies amb anterioritat mires abans de predir
    FORECAST_RANGE = 7                # quants dies em de predir
    n_features = len(cc_predictors)   # quantes variables (sistema molt sensible)
    
    # HYPERPARAMETERS ............................................
    epochs = 10
    batch_size = 128
    validation = 0.1
    patience = 10

    X_train, y_train = split_sequence(scaled_train, look_back=LOOK_BACK, forecast_horizon=FORECAST_RANGE)
    X_test, y_test = split_sequence(scaled_test, look_back=LOOK_BACK, forecast_horizon=FORECAST_RANGE)

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    # MODEL TRAINING --------------------------------------------------------
    # GRU model .........................................................................................................
    print('\n >>>>> model 1: GRU')
    model_gru = create_model_gru(X_train)
    history = fit_model(model_gru,X_train, y_train, epochs, batch_size, validation, patience)
    yhat_gru = prediction(model_gru,X_test)
    y_test_inverse, yhat_gru_inverse = inverse_transform(y_test, yhat_gru, scaler)
    plt_model(y_test_inverse, yhat_gru_inverse,"GRU")

    # LSTM model ........................................................................................................
    print('\n >>>>> model 2: LSTM')
    model_lstm = create_model_lstm(X_train)
    history = fit_model(model_lstm,X_train, y_train, epochs, batch_size, validation, patience)
    yhat_lstm = prediction(model_lstm,X_test)
    y_test_inverse, yhat_lstm_inverse = inverse_transform(y_test, yhat_lstm, scaler)
    plt_model(y_test_inverse, yhat_lstm_inverse,"LSTM")

    # Bi-directional model .............................................................................................
    print('\n >>>>> model 3: Bi-directional')
    model_bilstm = create_model_bilstm(X_train)
    history = fit_model(model_bilstm,X_train, y_train, epochs, batch_size, validation, patience)
    yhat_bilstm = prediction(model_bilstm,X_test)
    y_test_inverse, yhat_bilstm_inverse = inverse_transform(y_test, yhat_bilstm, scaler)
    plt_model(y_test_inverse, yhat_bilstm_inverse,"Bi-directional_LSTM")

    # Encoder-decoder LSTM model .......................................................................................
    print('\n >>>>> model 4: Encoder-decoder LSTM')
    model_enc_dec = create_model_enc_dec(X_train)
    history = fit_model(model_enc_dec,X_train, y_train, epochs, batch_size, validation, patience)
    yhat_endelstm = prediction(model_enc_dec,X_test)
    y_test_inverse, yhat_endelstm_inverse = inverse_transform(y_test, yhat_endelstm, scaler)
    plt_model(y_test_inverse, yhat_endelstm_inverse,"ENCODER_DECODER_LSTM")

    # CNN-LSTM Encoder-Decoder model ...................................................................................
    print('\n >>>>> model 5: CNN-LSTM Encoder-Decoder')
    model_enc_dec_cnn = create_model_enc_dec_cnn(X_train)
    history = fit_model(model_enc_dec_cnn, X_train, y_train, epochs, batch_size, validation, patience)
    yhat_cnnlstmende = prediction(model_enc_dec_cnn,X_test)
    y_test_inverse, yhat_cnnlstmende_inverse = inverse_transform(y_test, yhat_cnnlstmende, scaler)
    plt_model(y_test_inverse, yhat_cnnlstmende_inverse,"Encoder_DECODER_CNN_LSTM")

    # Vector-Output model .............................................................................................
    print('\n >>>>> model 6: Vector-Output')
    model_vector_output = create_model_vector_output(X_train)
    history = fit_model(model_vector_output, X_train, y_train, epochs, batch_size, validation, patience)
    yhat_veout = prediction(model_vector_output,X_test)
    y_test_inverse, yhat_veout_inverse = inverse_transform(y_test, yhat_veout, scaler)
    plt_model(y_test_inverse, yhat_veout_inverse,"Vector_Output")

    # Multi-Head CNN-LSTM model ......................................................................................
    print('\n >>>>> model 7: Multi-Head CNN-LSTM')
    multi_head_cnn_lstm_model = create_model_multi_head_cnn_lstm(X_train)
    history = fit_model(multi_head_cnn_lstm_model, X_train, y_train, epochs, batch_size, validation, patience)
    yhat_muhecnnlstm = prediction(multi_head_cnn_lstm_model,X_test)
    y_test_inverse, yhat_muhecnnlstm_inverse = inverse_transform(y_test, yhat_muhecnnlstm, scaler)
    plt_model(y_test_inverse, yhat_muhecnnlstm_inverse, "Multi-Head CNN-LSTM")  

    # MODEL EVALUATION --------------------------------------------------------
    print('\n >>>>> EVALUATION OF THE DIFFERENT MODELS')

    # GRU model .........................................................................................................
    print('\n >>>>> model 1: GRU')
    evaluate_forecast(y_test_inverse, yhat_gru_inverse)

    # LSTM model ........................................................................................................
    print('\n >>>>> model 2: LSTM')
    evaluate_forecast(y_test_inverse, yhat_lstm_inverse)

    # Bi-directional model .............................................................................................
    print('\n >>>>> model 3: Bi-directional')
    evaluate_forecast(y_test_inverse, yhat_bilstm_inverse)

    # Encoder-decoder LSTM model .......................................................................................
    print('\n >>>>> model 4: Encoder-decoder LSTM')
    evaluate_forecast(y_test_inverse, yhat_endelstm_inverse)

    # CNN-LSTM Encoder-Decoder model ...................................................................................
    print('\n >>>>> model 5: CNN-LSTM Encoder-Decoder')
    evaluate_forecast(y_test_inverse, yhat_cnnlstmende_inverse)

    # Vector-Output model .............................................................................................
    print('\n >>>>> model 6: Vector-Output')
    evaluate_forecast(y_test_inverse, yhat_veout_inverse)

    # Multi-Head CNN-LSTM model ......................................................................................
    print('\n >>>>> model 7: Multi-Head CNN-LSTM')
    evaluate_forecast(y_test_inverse, yhat_muhecnnlstm_inverse)

    # PLOT ----------------------------------------------------------------------------------------------------------
    # plot models
    plt.figure(figsize=(20, 10))
    plt.plot(pd.DataFrame(y_test_inverse)[[col_idx]], label='True Values', linewidth=2.5)
    plt.plot(pd.DataFrame(yhat_gru_inverse)[[col_idx]], label='Pred. GRU', linestyle='--')
    plt.plot(pd.DataFrame(yhat_lstm_inverse)[[col_idx]], label='Pred. LSTM', linestyle='--')
    plt.plot(pd.DataFrame(yhat_bilstm_inverse)[[col_idx]], label='Pred. BiLSTM', linestyle='--')
    plt.plot(pd.DataFrame(yhat_endelstm_inverse)[[col_idx]], label='Pred. Enc-Dec LSTM', linestyle='--')
    plt.plot(pd.DataFrame(yhat_cnnlstmende_inverse)[[col_idx]], label='Pred. CNN-LSTM Enc-Dec', linestyle='--')
    plt.plot(pd.DataFrame(yhat_veout_inverse)[[col_idx]], label='Pred. Vector Output', linestyle='--')
    plt.plot(pd.DataFrame(yhat_muhecnnlstm_inverse)[[col_idx]], label='Pred. Mult-Head CNN-LSTM', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Real vs. Predicted Values MODELS')
    plt.legend()
    plt.show()

    print("Deep learning models training completed successfully!")

