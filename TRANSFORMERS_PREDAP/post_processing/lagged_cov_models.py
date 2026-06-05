"""
==========================================================
Time Series Forecasting with Hybrid LSTM + Transformer Model
==========================================================

This script defines a deep learning model that combines Long Short-Term Memory (LSTM) 
networks and Transformer encoders to forecast demand in healthcare diagnostic visits.

Key Components:
- **Hybrid Model**: Uses LSTMs for sequential feature extraction and Transformers 
  for capturing long-range dependencies in time series data.
- **Transformer Encoder Block**: Implements multi-head self-attention and 
  a feed-forward network with residual connections.
- **Training Function**: Automates model training with GPU memory optimization, 
  early stopping, and model checkpointing.
- **Residual Learning**: Enhances forecasting accuracy by refining predictions 
  using seasonal patterns and lagged variables.

Usage:
- Import this module into a Jupyter Notebook or another script.
- Build the model, train it with your dataset, and evaluate performance.

Author: Guillem Hernández Guillamet
Date: 18/03/2025
Version: 1.0
"""


import os
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def transformer_encoder(
    inputs: tf.Tensor, 
    head_size: int, 
    num_heads: int, 
    ff_dim: int, 
    dropout: float = 0.2
) -> tf.Tensor:
    """
    Defines a Transformer Encoder block.

    Parameters:
    - inputs (tf.Tensor): Input tensor to the transformer encoder.
    - head_size (int): Dimensionality of each attention head.
    - num_heads (int): Number of attention heads.
    - ff_dim (int): Dimensionality of the feed-forward layer.
    - dropout (float, optional): Dropout rate. Default is 0.2.

    Returns:
    - tf.Tensor: Output tensor after applying transformer encoder block.
    """
    
    # Layer Normalization before Multi-Head Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs  # Residual connection

    # Feed Forward Network
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="tanh")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)

    return x + res  # Final residual connection

def build_hybrid_lstm_transformer(forecast_horizon, input_shape, transformer_encoder):
    """
    Builds and returns a hybrid LSTM + Transformer model for time series forecasting.

    Parameters:
    - forecast_horizon (int): Number of time steps to predict.
    - input_shape (tuple): Shape of the input data (time steps, features).
    - transformer_encoder (function): Transformer encoder function to apply.

    Returns:
    - keras.Model: Compiled hybrid LSTM + Transformer model.
    """

    # Define Input Layer
    input_layer = keras.Input(shape=(forecast_horizon, input_shape[-1]))

    # LSTM Block
    x = layers.LSTM(64, return_sequences=True)(input_layer)
    x = layers.Dropout(0.2)(x)  # Dropout to reduce overfitting
    x = layers.LSTM(32, return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)

    # Transformer Block
    x = transformer_encoder(x, head_size=2, num_heads=2, ff_dim=8, dropout=0.2)

    # Output Layer
    outputs = layers.TimeDistributed(layers.Dense(1))(x)

    # Build Model
    model = keras.Model(inputs=input_layer, outputs=outputs)

    # Compile Model
    model.compile(optimizer='adam', loss='mse')

    return model



def train_given_model_and_data(model, X, Y, batch_size=1024, model_name=None, epochs=100, 
                               save_history=False, save_model=True, save_memory=True, 
                               shuffle=False, callbacks=None):
    """
    Trains a given TensorFlow/Keras model with provided data and optional configurations.

    Parameters:
    - model (tf.keras.Model): The Keras model to be trained.
    - X (numpy.ndarray or tf.Tensor): Input training data.
    - Y (numpy.ndarray or tf.Tensor): Target values for training.
    - batch_size (int, optional): Batch size for training. Default is 1024.
    - model_name (str, optional): Name of the model for saving purposes. If None, it defaults to 'testing'.
    - epochs (int, optional): Number of training epochs. Default is 100.
    - save_history (bool, optional): Whether to save training history. Default is False.
    - save_model (bool, optional): Whether to save the trained model. Default is True.
    - save_memory (bool, optional): If True, configures GPU memory growth and logs memory usage. Default is True.
    - shuffle (bool, optional): Whether to shuffle data during training. Default is False.
    - callbacks (list, optional): List of Keras callbacks. If None, Early Stopping is used by default.

    Returns:
    - None: The function trains the model and optionally saves history and memory usage.
    """

    # Configure GPU memory growth to prevent memory overflow
    if save_memory:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"GPU memory configuration error: {e}")

    # Default callback setup (if none provided)
    if callbacks is None:
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', mode='min', patience=25, restore_best_weights=True
        )
        callbacks = [early_stop]

    # If a model name is provided, check if it already exists
    if model_name and os.path.exists(model_name):
        print(f"Model '{model_name}' already exists. Skipping training.")
        return

    # Train the model
    history = model.fit(
        x=X, 
        y=Y, 
        batch_size=batch_size,  
        epochs=epochs, 
        shuffle=shuffle,        
        validation_split=0.1,   
        callbacks=callbacks
    )

    # Set default model name if none is provided
    if model_name is None:
        model_name = "testing"

    # Save training history if requested
    if save_history:
        with open(f'{model_name}_history.pkl', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

    # Save trained model if requested
    if save_model and epochs > 1:
        model.save(model_name)

    # Log GPU memory usage if save_memory is enabled
    if save_memory:
        try:
            memory_info = tf.config.experimental.get_memory_info('GPU:0')
            with open('memory.csv', 'a') as resultcsv:
                resultcsv.write(f"{model_name},{memory_info['peak']},train\n")
            print(f"Current memory usage: {memory_info['current'] / (1024**2)} MB")
            print(f"Peak memory usage: {memory_info['peak'] / (1024**2)} MB")
        except Exception as e:
            print(f"Memory logging error: {e}")