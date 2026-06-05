# models_dl.py
# =============
# Neural network model definitions for time series prediction
# Author: Guillem Hernández Guillamet
# Version: 1.0

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    GRU, LSTM, Dense, Dropout, RepeatVector, TimeDistributed, 
    Bidirectional, Conv1D, MaxPooling1D, Flatten, Input, 
    Reshape, Concatenate
)


def create_model_gru(X_train, optimizer='adam', forecast_range=None, n_features=None):
    """
    Create a GRU-based sequence-to-sequence model.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training data to determine input shape
    optimizer : str
        Optimizer for model compilation
    forecast_range : int
        Number of time steps to forecast
    n_features : int
        Number of features in the data
        
    Returns:
    --------
    tensorflow.keras.Model
        Compiled GRU model
    """
    try:
        # Use global values if not provided
        if forecast_range is None:
            forecast_range = globals().get('FORECAST_RANGE', 7)
        if n_features is None:
            n_features = globals().get('n_features', X_train.shape[2])
            
        model = Sequential()
        model.add(GRU(units=100, input_shape=[X_train.shape[1], X_train.shape[2]]))
        model.add(RepeatVector(forecast_range))
        model.add(Dropout(0.2))
        model.add(GRU(units=100, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(TimeDistributed(Dense(n_features)))
        
        model.compile(loss='mse', optimizer=optimizer)
        return model
        
    except Exception as e:
        print(f"Error creating GRU model: {str(e)}")
        return None


def create_model_lstm(X_train, optimizer='adam', forecast_range=None, n_features=None):
    """
    Create an LSTM-based sequence-to-sequence model.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training data to determine input shape
    optimizer : str
        Optimizer for model compilation
    forecast_range : int
        Number of time steps to forecast
    n_features : int
        Number of features in the data
        
    Returns:
    --------
    tensorflow.keras.Model
        Compiled LSTM model
    """
    try:
        # Use global values if not provided
        if forecast_range is None:
            forecast_range = globals().get('FORECAST_RANGE', 7)
        if n_features is None:
            n_features = globals().get('n_features', X_train.shape[2])
            
        model = Sequential()
        model.add(LSTM(units=100, input_shape=[X_train.shape[1], X_train.shape[2]]))
        model.add(RepeatVector(forecast_range))
        model.add(Dropout(0.2))
        model.add(LSTM(units=100, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(TimeDistributed(Dense(n_features)))
        
        model.compile(loss='mse', optimizer=optimizer)
        return model
        
    except Exception as e:
        print(f"Error creating LSTM model: {str(e)}")
        return None


def create_model_bilstm(X_train, optimizer='adam', forecast_range=None, n_features=None):
    """
    Create a Bidirectional LSTM model.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training data to determine input shape
    optimizer : str
        Optimizer for model compilation
    forecast_range : int
        Number of time steps to forecast
    n_features : int
        Number of features in the data
        
    Returns:
    --------
    tensorflow.keras.Model
        Compiled Bidirectional LSTM model
    """
    try:
        # Use global values if not provided
        if forecast_range is None:
            forecast_range = globals().get('FORECAST_RANGE', 7)
        if n_features is None:
            n_features = globals().get('n_features', X_train.shape[2])
            
        model = Sequential()
        model.add(Bidirectional(LSTM(units=100), input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(RepeatVector(forecast_range))
        model.add(Bidirectional(LSTM(units=100, return_sequences=True)))
        model.add(TimeDistributed(Dense(n_features)))
        
        model.compile(loss='mse', optimizer=optimizer)
        return model
        
    except Exception as e:
        print(f"Error creating Bidirectional LSTM model: {str(e)}")
        return None


def create_model_enc_dec(X_train, optimizer='adam', forecast_range=None, n_features=None):
    """
    Create an Encoder-Decoder LSTM model.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training data to determine input shape
    optimizer : str
        Optimizer for model compilation
    forecast_range : int
        Number of time steps to forecast
    n_features : int
        Number of features in the data
        
    Returns:
    --------
    tensorflow.keras.Model
        Compiled Encoder-Decoder LSTM model
    """
    try:
        # Use global values if not provided
        if forecast_range is None:
            forecast_range = globals().get('FORECAST_RANGE', 7)
        if n_features is None:
            n_features = globals().get('n_features', X_train.shape[2])
            
        model_enc_dec = Sequential()
        model_enc_dec.add(LSTM(100, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
        model_enc_dec.add(RepeatVector(forecast_range))
        model_enc_dec.add(LSTM(100, activation='relu', return_sequences=True))
        model_enc_dec.add(TimeDistributed(Dense(n_features)))
        
        model_enc_dec.compile(optimizer=optimizer, loss='mse')
        return model_enc_dec
        
    except Exception as e:
        print(f"Error creating Encoder-Decoder LSTM model: {str(e)}")
        return None


def create_model_enc_dec_cnn(X_train, optimizer='adam', kern_size=3, forecast_range=None, n_features=None):
    """
    Create a CNN-LSTM Encoder-Decoder model.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training data to determine input shape
    optimizer : str
        Optimizer for model compilation
    kern_size : int
        Kernel size for convolutional layers
    forecast_range : int
        Number of time steps to forecast
    n_features : int
        Number of features in the data
        
    Returns:
    --------
    tensorflow.keras.Model
        Compiled CNN-LSTM Encoder-Decoder model
    """
    try:
        # Use global values if not provided
        if forecast_range is None:
            forecast_range = globals().get('FORECAST_RANGE', 7)
        if n_features is None:
            n_features = globals().get('n_features', X_train.shape[2])
            
        model_enc_dec_cnn = Sequential()
        model_enc_dec_cnn.add(Conv1D(filters=64, kernel_size=kern_size, activation='relu', 
                                     input_shape=(X_train.shape[1], X_train.shape[2])))
        model_enc_dec_cnn.add(Conv1D(filters=64, kernel_size=kern_size, activation='relu'))
        model_enc_dec_cnn.add(MaxPooling1D(pool_size=2))
        model_enc_dec_cnn.add(Flatten())
        model_enc_dec_cnn.add(RepeatVector(forecast_range))
        model_enc_dec_cnn.add(LSTM(200, activation='relu', return_sequences=True))
        model_enc_dec_cnn.add(TimeDistributed(Dense(100, activation='relu')))
        model_enc_dec_cnn.add(TimeDistributed(Dense(n_features)))
        
        model_enc_dec_cnn.compile(loss='mse', optimizer=optimizer)
        return model_enc_dec_cnn
        
    except Exception as e:
        print(f"Error creating CNN-LSTM Encoder-Decoder model: {str(e)}")
        return None


def create_model_vector_output(X_train, optimizer='adam', kern_size=3, forecast_range=None, n_features=None):
    """
    Create a Vector Output model using CNN and LSTM layers.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training data to determine input shape
    optimizer : str
        Optimizer for model compilation
    kern_size : int
        Kernel size for convolutional layers
    forecast_range : int
        Number of time steps to forecast
    n_features : int
        Number of features in the data
        
    Returns:
    --------
    tensorflow.keras.Model
        Compiled Vector Output model
    """
    try:
        # Use global values if not provided
        if forecast_range is None:
            forecast_range = globals().get('FORECAST_RANGE', 7)
        if n_features is None:
            n_features = globals().get('n_features', X_train.shape[2])
            
        input_layer = Input(shape=(X_train.shape[1], X_train.shape[2])) 
        conv = Conv1D(filters=4, kernel_size=kern_size, activation='relu')(input_layer)
        conv = Conv1D(filters=6, kernel_size=kern_size, activation='relu')(conv)

        lstm = LSTM(100, return_sequences=True, activation='relu')(conv)
        dropout = Dropout(0.2)(lstm)
        lstm = LSTM(100, activation='relu')(dropout)
        dense = Dense(forecast_range * n_features, activation='relu')(lstm)
        output_layer = Reshape((forecast_range, n_features))(dense)
        model_vector_output = Model([input_layer], [output_layer])
        
        model_vector_output.compile(optimizer=optimizer, loss='mse')
        return model_vector_output
        
    except Exception as e:
        print(f"Error creating Vector Output model: {str(e)}")
        return None


def create_model_multi_head_cnn_lstm(X_train, optimizer='adam', kern_size=3, forecast_range=None, n_features=None):
    """
    Create a Multi-Head CNN-LSTM model.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training data to determine input shape
    optimizer : str
        Optimizer for model compilation
    kern_size : int
        Kernel size for convolutional layers
    forecast_range : int
        Number of time steps to forecast
    n_features : int
        Number of features in the data
        
    Returns:
    --------
    tensorflow.keras.Model
        Compiled Multi-Head CNN-LSTM model
    """
    try:
        # Use global values if not provided
        if forecast_range is None:
            forecast_range = globals().get('FORECAST_RANGE', 7)
        if n_features is None:
            n_features = globals().get('n_features', X_train.shape[2])
            
        input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))  # Look back, n_features
        head_list = []
        
        for i in range(0, n_features):
            conv_layer_head = Conv1D(filters=4, kernel_size=kern_size, activation='relu')(input_layer)
            conv_layer_head_2 = Conv1D(filters=6, kernel_size=kern_size, activation='relu')(conv_layer_head)
            conv_layer_flatten = Flatten()(conv_layer_head_2)
            head_list.append(conv_layer_flatten)

        concat_cnn = Concatenate(axis=1)(head_list)
        reshape = Reshape((head_list[0].shape[1], n_features))(concat_cnn)
        lstm = LSTM(100, activation='relu')(reshape)
        repeat = RepeatVector(forecast_range)(lstm)
        lstm_2 = LSTM(100, activation='relu', return_sequences=True)(repeat)
        dropout = Dropout(0.2)(lstm_2)
        dense = Dense(n_features, activation='linear')(dropout)
        multi_head_cnn_lstm_model = Model(inputs=input_layer, outputs=dense)
        
        multi_head_cnn_lstm_model.compile(optimizer=optimizer, loss='mse')
        return multi_head_cnn_lstm_model
        
    except Exception as e:
        print(f"Error creating Multi-Head CNN-LSTM model: {str(e)}")
        return None