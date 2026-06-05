# Deep Learning Module API Reference

The Deep Learning module provides neural network architectures and training utilities for time series forecasting.

## **Overview**

The Deep Learning module consists of several submodules:

- **`models_dl`**: Neural network model architectures (GRU, LSTM, BiLSTM, CNN-LSTM)
- **`training_dl`**: Model training and optimization procedures
- **`prediction_dl`**: Inference and prediction utilities
- **`preprocessing_dl`**: Data preprocessing for deep learning models
- **`evaluation_dl`**: Model evaluation and performance metrics
- **`visualization_dl`**: Training progress and results visualization

## Model Architectures

The module supports various neural network architectures optimized for time series forecasting:

### GRU Models
- **create_model_gru**: Creates Gated Recurrent Unit models for sequence prediction
- Suitable for medium-length sequences with good computational efficiency

### LSTM Models  
- **create_model_lstm**: Creates Long Short-Term Memory models
- **create_model_bilstm**: Creates bidirectional LSTM models
- Ideal for capturing long-term dependencies in time series

### CNN-LSTM Hybrid Models
- **create_model_enc_dec_cnn**: CNN-based encoder-decoder architecture
- **create_model_multi_head_cnn_lstm**: Multi-head CNN-LSTM combination
- Combines spatial feature extraction with temporal modeling

### Advanced Architectures
- **create_model_enc_dec**: Encoder-decoder models for sequence-to-sequence tasks
- **create_model_vector_output**: Models with vector output for multi-step forecasting

## Training Pipeline

The training module provides comprehensive model training capabilities:

- **Model Compilation**: Automatic model compilation with appropriate loss functions
- **Training Procedures**: Batch training with validation monitoring
- **Early Stopping**: Automatic training termination to prevent overfitting
- **Model Checkpointing**: Save best models during training

## Prediction Engine

Generate forecasts from trained models:

- **Forward Prediction**: Generate future predictions from trained models
- **Inverse Transformation**: Convert scaled predictions back to original scale
- **Batch Prediction**: Efficient prediction for multiple sequences

## Data Preprocessing

Specialized preprocessing for deep learning models:

- **Sequence Splitting**: Convert time series to supervised learning format
- **Temporality Addition**: Add temporal features to enhance model performance
- **Scaling and Normalization**: Prepare data for neural network training

## Evaluation

::: src.dl.evaluation_dl.evaluate_forecast

## Visualization

::: src.dl.visualization_dl.plot_train_test

::: src.dl.visualization_dl.plt_model

## Usage Examples

### Basic LSTM Model

```python
from src.dl import (
    split_sequence, create_model_lstm, fit_model,
    prediction, inverse_transform, evaluate_forecast
)
from sklearn.preprocessing import MinMaxScaler

# Prepare data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Create sequences
LOOK_BACK = 30
FORECAST_HORIZON = 7

X_train, y_train = split_sequence(
    scaled_data[:800], 
    look_back=LOOK_BACK, 
    forecast_horizon=FORECAST_HORIZON
)
X_test, y_test = split_sequence(
    scaled_data[800:], 
    look_back=LOOK_BACK, 
    forecast_horizon=FORECAST_HORIZON
)

# Create and train model
model = create_model_lstm(X_train)
trained_model = fit_model(
    model, 
    X_train, 
    y_train,
    epochs=100,
    batch_size=32,
    validation=0.2,
    patience=10
)

# Make predictions
predictions = prediction(trained_model, X_test)

# Inverse transform and evaluate
y_test_orig, pred_orig = inverse_transform(y_test, predictions, scaler)
metrics = evaluate_forecast(y_test_orig, pred_orig)
```

### Multi-Model Comparison

```python
from src.dl import (
    create_model_gru, create_model_lstm, create_model_bilstm,
    create_model_enc_dec, plt_model
)

# Define models to compare
models_config = {
    'GRU': create_model_gru,
    'LSTM': create_model_lstm,
    'BiLSTM': create_model_bilstm,
    'Encoder-Decoder': create_model_enc_dec
}

results = {}

for name, model_func in models_config.items():
    print(f"Training {name}...")
    
    # Create and train model
    model = model_func(X_train)
    trained_model = fit_model(model, X_train, y_train, epochs=50)
    
    # Predict and evaluate
    pred = prediction(trained_model, X_test)
    _, pred_orig = inverse_transform(y_test, pred, scaler)
    
    # Store results
    results[name] = {
        'model': trained_model,
        'predictions': pred_orig,
        'metrics': evaluate_forecast(y_test_orig, pred_orig)
    }
    
    # Plot results
    plt_model(y_test_orig, pred_orig, name)
```

### Advanced CNN-LSTM Model

```python
from src.dl import create_model_multi_head_cnn_lstm

# Create advanced architecture
model = create_model_multi_head_cnn_lstm(X_train)

# Custom training configuration
history = fit_model(
    model,
    X_train,
    y_train,
    epochs=200,
    batch_size=64,
    validation=0.15,
    patience=20,
    learning_rate=0.001
)

# Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.legend()
plt.show()
```

## Model Architectures Details

### GRU (Gated Recurrent Unit)

**Use Case**: General-purpose RNN for time series with moderate complexity

**Architecture**:
- Input layer
- GRU layer with dropout
- Dense output layer

**Parameters**:
- Units: 50 (default)
- Dropout: 0.2
- Return sequences: True for multi-step forecasting

### LSTM (Long Short-Term Memory)

**Use Case**: Complex temporal patterns with long-term dependencies

**Architecture**:
- Input layer
- LSTM layer with dropout and recurrent dropout
- Dense output layer

**Parameters**:
- Units: 50 (default)
- Dropout: 0.2
- Recurrent dropout: 0.2

### BiLSTM (Bidirectional LSTM)

**Use Case**: When both past and future context is important

**Architecture**:
- Input layer
- Bidirectional LSTM wrapper
- Dense output layer

**Benefits**: Processes sequences in both directions

### Encoder-Decoder LSTM

**Use Case**: Sequence-to-sequence forecasting

**Architecture**:
- Encoder LSTM (returns states)
- Decoder LSTM (uses encoder states)
- TimeDistributed Dense layer

### CNN-LSTM Hybrid

**Use Case**: Complex patterns with both spatial and temporal features

**Architecture**:
- 1D Convolutional layers
- MaxPooling layers
- LSTM layers
- Dense output layer

## Training Configuration

### Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs` | int | 100 | Number of training epochs |
| `batch_size` | int | 32 | Training batch size |
| `validation` | float | 0.2 | Validation split ratio |
| `patience` | int | 10 | Early stopping patience |
| `learning_rate` | float | 0.001 | Optimizer learning rate |

### Hyperparameter Tuning

```python
from itertools import product

# Define hyperparameter grid
param_grid = {
    'epochs': [50, 100, 200],
    'batch_size': [16, 32, 64],
    'units': [32, 50, 100],
    'dropout': [0.1, 0.2, 0.3]
}

best_score = float('inf')
best_params = None

for params in product(*param_grid.values()):
    # Unpack parameters
    epochs, batch_size, units, dropout = params
    
    # Create model with parameters
    model = create_model_lstm(X_train, units=units, dropout=dropout)
    
    # Train model
    trained_model = fit_model(
        model, X_train, y_train,
        epochs=epochs, batch_size=batch_size,
        validation=0.2, verbose=0
    )
    
    # Evaluate
    pred = prediction(trained_model, X_test)
    _, pred_orig = inverse_transform(y_test, pred, scaler)
    metrics = evaluate_forecast(y_test_orig, pred_orig)
    
    # Check if best
    if metrics['rmse'] < best_score:
        best_score = metrics['rmse']
        best_params = params

print(f"Best parameters: {best_params}")
print(f"Best RMSE: {best_score}")
```

## Data Preprocessing

### Sequence Creation

The `split_sequence` function creates input-output pairs for supervised learning:

```python
# For multi-step forecasting
X, y = split_sequence(data, look_back=30, forecast_horizon=7)
print(f"Input shape: {X.shape}")   # (samples, timesteps, features)
print(f"Output shape: {y.shape}")  # (samples, forecast_horizon, features)
```

### Temporal Features

Add time-based features like day of week, month:

```python
from src.dl import add_temprality

# Add temporal features
df_with_time, updated_columns = add_temprality(df, df.columns.tolist())
print(f"Added temporal features: {set(updated_columns) - set(df.columns)}")
```

## Evaluation Metrics

The `evaluate_forecast` function returns:

- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error  
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of determination
- **Directional Accuracy**: Percentage of correct direction predictions

## Performance Optimization

### GPU Acceleration

```python
import tensorflow as tf

# Check GPU availability
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

### Memory Management

```python
# For large datasets, use generators
def data_generator(X, y, batch_size):
    while True:
        for i in range(0, len(X), batch_size):
            yield X[i:i+batch_size], y[i:i+batch_size]

# Train with generator
model.fit(
    data_generator(X_train, y_train, batch_size=32),
    steps_per_epoch=len(X_train) // 32,
    epochs=100,
    validation_data=(X_val, y_val)
)
```

## Error Handling

Common issues and solutions:

- **Shape Mismatch**: Ensure input shapes match model expectations
- **Memory Issues**: Reduce batch size or use data generators
- **Convergence Problems**: Adjust learning rate or model architecture
- **Overfitting**: Increase dropout, add regularization, or use early stopping

## Related Functions

See also:

- [LMLR Module](lmlr.md) for feature selection
- [Granger Causality Module](gcausal.md) for causal feature selection
- [Utilities](utilities.md) for data preprocessing helpers