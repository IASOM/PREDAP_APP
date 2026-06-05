"""
Model Architecture Module for Residual Multivariate Transformers
================================================================

This module contains model architectures for residual multivariate transformer models,
including transformer encoders, hybrid LSTM-transformer models, and learning rate schedules.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import MaxNorm

from config.config_manager import get_config
from model_architechture.layers import CustomCosineDecay, PositionalEncoding, RevIN
from model_architechture.transformer_univ_architechtures.model_architechture_informer import build_informer_model
from model_architechture.transformer_univ_architechtures.model_architechture_log_transformer import build_log_transformer_model
from model_architechture.transformer_univ_architechtures.model_architechture_LSTNet import build_lstnet_model
from model_architechture.transformer_univ_architechtures.model_architechture_base_tranformer import build_base_model

default_config = get_config()

def transformer_encoder(inputs, head_size=None, num_heads=None, ff_dim=None, dropout=None, activation_function='tanh'):
    """
    Transformer Encoder Block for processing sequential data.
    
    Parameters:
    -----------
    inputs : tf.Tensor
        Input tensor with shape (batch_size, sequence_length, features)
    head_size : int, optional
        Dimensions of each attention head (default from config)
    num_heads : int, optional
        Number of attention heads (default from config)
    ff_dim : int, optional
        Dimensionality of feed-forward layer (default from config)
    dropout : float, optional
        Dropout rate (default from config)
        
    Returns:
    --------
    tf.Tensor
        Processed tensor with same shape as input
    """
    # Use default parameters if not provided
    if head_size is None:
        head_size = default_config.DEFAULT_RESIDUAL_TRANSFORMER_PARAMS['head_size']
    if num_heads is None:
        num_heads = default_config.DEFAULT_RESIDUAL_TRANSFORMER_PARAMS['num_heads']
    if ff_dim is None:
        ff_dim = default_config.DEFAULT_RESIDUAL_TRANSFORMER_PARAMS['ff_dim']
    if dropout is None:
        dropout = default_config.DEFAULT_RESIDUAL_TRANSFORMER_PARAMS['dropout']
    
    # Multi-Head Self-Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)                  
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)  
    x = layers.Dropout(dropout)(x) 
    res = x + inputs  

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)                       
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation=activation_function)(x) 
    x = layers.Dropout(dropout)(x)                                         
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)          
    return x + res    

    
def hybrid_lstm_transformer_model(input_shape, forecast, 
                                    lstm_params=None, 
                                transformer_params=None,
                                activation_function='tanh'):
    """
    Build a hybrid LSTM-Transformer model for residual learning.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (sequence_length, features)
    forecast : int
        Number of future time steps to predict
    lstm_params : dict, optional
        LSTM layer parameters (default from config)
    transformer_params : dict, optional
        Transformer encoder parameters (default from config)
        
    Returns:
    --------
    tf.keras.Model
        Compiled hybrid LSTM-Transformer model
    """
    # Use default parameters if not provided
    if lstm_params is None:
        lstm_params = default_config.DEFAULT_RESIDUAL_LSTM_PARAMS.copy()
    if transformer_params is None:
        transformer_params = default_config.DEFAULT_RESIDUAL_TRANSFORMER_PARAMS.copy()


    
    input_layer = keras.Input(shape=input_shape)
    revin_layer = RevIN()
    x, mean, stdev = revin_layer(input_layer, mode='norm')
    
    d_model = max(transformer_params['head_size'] * transformer_params['num_heads'], 32)
    x = layers.Dense(d_model)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    # Transformer Block
    x = PositionalEncoding(input_shape[0], d_model)(x)

    for _ in range(transformer_params['num_transformer_blocks']):
        x = transformer_encoder(
            x, 
            head_size=transformer_params['head_size'],
            num_heads=transformer_params['num_heads'],
            ff_dim=transformer_params['ff_dim'],
            activation_function=activation_function,
            dropout=transformer_params['dropout']
        )

    # GlobalAveragePooling1D Layer
    #x = layers.GlobalAveragePooling1D(data_format="channels_last")(x) # May be changed to Flatten() if needed 
    if input_shape[0] >= 60:  # Only apply pooling if sequence length is sufficient
        x = layers.AveragePooling1D(30, data_format="channels_first")(x)
    x = layers.Flatten()(x)
    
    for dim in transformer_params['mlp_units']:
        x = layers.Dense(dim, activation=activation_function)(x)
        x = layers.Dropout(transformer_params['dropout'])(x)
    # Output Layer
    outputs = layers.Dense(forecast, activation='linear')(x)

    # Reshape Outputs
    outputs = layers.Reshape((forecast, 1))(outputs)
    outputs = revin_layer(outputs, mode='denorm', mean=mean, stdev=stdev)
    outputs = layers.Reshape((forecast,))(outputs)


    # Build Model
    model = keras.Model(inputs=input_layer, outputs=outputs)

    # Compile Model
    #model.compile(optimizer='adam', loss='mse')

    return model
    
def build_residual_transformer_model(input_shape, forecast,
                                    lstm_params=None, 
                                transformer_params=None,
                                activation_function='tanh'):
    """
    Build a residual transformer model for time series forecasting.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (sequence_length, features)
    forecast : int
        Number of future time steps to predict
    lstm_params : dict, optional
        LSTM layer parameters (default from config)
    transformer_params : dict, optional
        Transformer encoder parameters (default from config)
        
    Returns:
    --------
    tf.keras.Model
        Compiled residual transformer model
    """
    return hybrid_lstm_transformer_model(input_shape, forecast, lstm_params, transformer_params, activation_function)    

'''def hybrid_lstm_transformer_model(input_shape, forecast, 
                                  lstm_params=None, 
                                  transformer_params=None,
                                  activation_function='tanh'):
    if lstm_params is None:
        lstm_params = default_config.DEFAULT_RESIDUAL_LSTM_PARAMS.copy()
    if transformer_params is None:
        transformer_params = default_config.DEFAULT_RESIDUAL_TRANSFORMER_PARAMS.copy()

    #return build_informer_model(input_shape, transformer_params['head_size'], transformer_params['num_heads'], transformer_params['ff_dim'], transformer_params['num_transformer_blocks'], transformer_params['mlp_units'], activation_function, transformer_params['dropout'], transformer_params['dropout'], forecast, pos_encoding=True)
    return build_log_transformer_model(input_shape, transformer_params['head_size'], transformer_params['num_heads'], transformer_params['ff_dim'], transformer_params['num_transformer_blocks'], transformer_params['mlp_units'], activation_function, transformer_params['dropout'], transformer_params['dropout'], forecast, pos_encoding=True)
    #return build_lstnet_model(input_shape, n_filters=transformer_params['head_size']*transformer_params['num_heads'], kernel_size=6, rnn_units=transformer_params['ff_dim'], skip_units=transformer_params['ff_dim']//2, skip=7, n_pred=forecast, dropout=transformer_params['dropout'])
    #return build_base_model(input_shape, transformer_params['head_size'], transformer_params['num_heads'], transformer_params['ff_dim'], transformer_params['num_transformer_blocks'], transformer_params['mlp_units'], activation_function, transformer_params['dropout'], transformer_params['dropout'], forecast, pos_encoding=True)
    '''

# shared layers (CustomCosineDecay, PositionalEncoding, RevIN) imported from src.core.layers

