# Residual Multivariate Transformers

This module provides a comprehensive framework for implementing residual correction using multivariate transformer models. It is designed to improve forecast accuracy by learning and correcting the residual errors from base prediction models.

## Overview

The residual multivariate transformer approach works in two stages:
1. **Base Model Predictions**: Use a pre-trained univariate transformer to generate initial forecasts
2. **Residual Correction**: Train a hybrid LSTM-Transformer model with multivariate covariates to predict and correct the residual errors

## Module Structure

The module follows a clean, modular architecture similar to the `univariate_transformer` package:

```
residual_multivariate_transformers/
├── __init__.py                                    # Package initialization and imports
├── config_residual_transformer.py                # Configuration parameters
├── model_architecture_residual_transformer.py    # Model architectures and learning rate schedules
├── training_evaluation_residual_transformer.py   # Training and evaluation utilities
├── utils_residual_transformer.py                 # Data processing and utility functions  
├── visualization_residual_transformer.py         # Plotting and visualization functions
├── main_training_residual_transformer.py         # Main training pipeline script
└── README.md                                      # This documentation file
```

## Key Components

### 1. Configuration (`config_residual_transformer.py`)
- Default parameters for data processing, model architecture, and training
- Pandemic wave definitions for evaluation
- File paths and naming conventions

### 2. Model Architecture (`model_architecture_residual_transformer.py`)
- **`transformer_encoder`**: Multi-head attention transformer encoder block
- **`hybrid_lstm_transformer_model`**: Hybrid LSTM-Transformer architecture for residual learning
- **`CustomCosineDecay`**: Learning rate scheduler with warmup and cosine decay

### 3. Training and Evaluation (`training_evaluation_residual_transformer.py`)
- **`train_given_model_and_data`**: Complete training pipeline with callbacks and GPU management
- **`evaluate_model`**: Model evaluation utilities
- **`setup_gpu_memory`**: GPU memory configuration
- **`create_model_directories`**: Directory structure setup

### 4. Utilities (`utils_residual_transformer.py`)
- **`split_train_test`**: Train-test data splitting with date filtering and normalization
- **`learn_covariates`**: Time series feature extraction and covariate preparation
- **`prepare_residual_data`**: Residual computation and analysis
- **`load_and_preprocess_data`**: Data loading and preprocessing pipeline

### 5. Visualization (`visualization_residual_transformer.py`)
- **`plot_residuals_analysis`**: Comprehensive residual analysis plots
- **`plot_predictions_with_pandemic_waves`**: Predictions overlaid with pandemic periods
- **`plot_stepwise_errors_comparison`**: Error comparison between original and corrected predictions
- **`evaluate_error_significance_pandemic_waves`**: Statistical analysis during pandemic waves

## Usage

### Basic Usage

```python
from residual_multivariate_transformers import (
    hybrid_lstm_transformer_model,
    train_given_model_and_data,
    split_train_test,
    learn_covariates
)

# 1. Load and prepare data
train_df, test_df = split_train_test(data)
df_processed = learn_covariates(train_df)

# 2. Build residual correction model
model = hybrid_lstm_transformer_model((lookback, num_features), forecast)

# 3. Train the model
train_given_model_and_data(model, X_train_covs, Y_residuals, 
                          model_name="residual_model")
```

### Complete Pipeline

Run the complete residual correction pipeline:

```python
# From the src directory
python -m residual_multivariate_transformers.main_training_residual_transformer
```

Or run the script directly:

```python
python residual_multivariate_transformers/main_training_residual_transformer.py
```

## Model Architecture Details

### Hybrid LSTM-Transformer Model

The residual correction model combines:

1. **LSTM Layers**: Capture temporal dependencies in the covariate data
   - Two LSTM layers (64 and 32 units) with dropout for regularization
   
2. **Transformer Encoder**: Apply self-attention to learned representations
   - Multi-head attention with configurable heads and dimensions
   - Layer normalization and residual connections
   - Feed-forward networks with tanh activation
   
3. **Output Layer**: TimeDistributed dense layer for multi-step predictions

### Learning Rate Schedule

Uses a custom cosine decay schedule with warmup:
- **Warmup Phase**: Linear increase from initial_lr to max_lr
- **Decay Phase**: Cosine decay from max_lr to min_lr

## Data Requirements

### Input Data Format
- CSV file with `timestamp` column and target diagnostic codes
- Data filtered to start after a specified date (default: 2010-01-01)
- Automatic min-max normalization applied to target variables

### Covariate Features
The model automatically generates time-based covariates:
- `Day_of_Week`: Day of the week (0-6)
- `Month`: Month of the year (1-12) 
- `Season`: Season (Spring, Summer, Fall, Winter)
- `Holiday`: Binary indicator for holidays
- `School_Vacation`: Binary indicator for school vacation periods

## Configuration Options

Key parameters can be customized in `config_residual_transformer.py`:

```python
# Model Architecture
DEFAULT_TRANSFORMER_PARAMS = {
    'head_size': 2,      # Attention head dimensions
    'num_heads': 2,      # Number of attention heads  
    'ff_dim': 8,         # Feed-forward dimension
    'dropout': 0.2       # Dropout rate
}

# Training Parameters
DEFAULT_TRAINING_PARAMS = {
    'batch_size': 32,        # Training batch size
    'epochs': 100,           # Maximum epochs
    'validation_split': 0.1,  # Validation data fraction
    'patience': 25           # Early stopping patience
}
```

## Output and Results

The pipeline generates:

1. **Trained Models**: Saved residual correction models
2. **Performance Metrics**: MAE, MSE, RMSE comparisons
3. **Visualizations**:
   - Residual analysis plots
   - Predictions with pandemic wave overlays
   - Error significance analysis
   - Training history plots

### Example Output
```
PERFORMANCE COMPARISON:
----------------------------------------
Original Model:
  MAE:  0.045123
  MSE:  0.003456  
  RMSE: 0.058789

Residual Corrected Model:
  MAE:  0.037891
  MSE:  0.002834
  RMSE: 0.053243

IMPROVEMENT:
  MAE:  +16.04%
  MSE:  +18.01%
  RMSE: +9.44%
```

## Dependencies

- TensorFlow 2.x
- pandas
- numpy  
- matplotlib
- scikit-learn
- Custom modules: `data_preparation`, `evaluation_plot_utils`

## Integration with Existing Codebase

This module is designed to work seamlessly with the existing TRANSFORMERS_PREDAP codebase:

- Follows the same naming conventions and structure as `univariate_transformer`
- Compatible with existing `data_preparation` and `evaluation_plot_utils` modules
- Uses the same model saving/loading patterns
- Integrates with the existing pandemic wave analysis framework

## Future Enhancements

Potential areas for extension:
- Support for additional covariate types (economic indicators, weather data)
- Ensemble residual correction methods
- Multi-target residual correction
- Real-time residual correction pipelines
- Integration with MLflow for experiment tracking

## Troubleshooting

### Common Issues

1. **GPU Memory Errors**: Reduce batch size or enable `save_memory=True`
2. **Missing Base Model**: Ensure the base transformer model exists in the models directory
3. **Data Shape Mismatches**: Check that lookback and forecast parameters match between base and residual models
4. **Import Errors**: Ensure all dependencies are installed and paths are correct

### Performance Tips

- Use GPU acceleration for faster training
- Experiment with different covariate combinations
- Tune hyperparameters using the configuration system
- Monitor validation loss for optimal early stopping

---

**Created by**: TRANSFORMERS_PREDAP Team  
**Version**: 1.0.0  
**Last Updated**: October 2025