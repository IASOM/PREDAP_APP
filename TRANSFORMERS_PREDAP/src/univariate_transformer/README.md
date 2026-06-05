# Univariate Transformer Module

A comprehensive, well-organized module for univariate time series forecasting using transformer models.

## 📁 File Structure

```
src/univariate_transformer/
├── __init__.py                              # Package initialization
├── config_univ_transformer.py              # Configuration and parameters
├── model_architecture_univ_transformer.py  # Transformer model architecture
├── training_evaluation_univ_transformer.py # Training and evaluation functions
├── visualization_univ_transformer.py       # Plotting and visualization
├── utils_univ_transformer.py              # Utility functions
├── main_training_univ_transformer.py      # Main pipeline script
└── README.md                              # This file
```

## 📋 Module Descriptions

### 1. `config_univ_transformer.py`
- **Purpose**: Centralized configuration management
- **Contains**: 
  - `TransformerConfig` class with all model parameters
  - Pandemic wave definitions
  - Hyperparameter search lists
  - Configuration utilities

### 2. `model_architecture_univ_transformer.py`
- **Purpose**: Transformer model architecture components
- **Contains**:
  - `transformer_encoder()`: Multi-head attention encoder block
  - `build_model()`: Complete model builder
  - `CustomCosineDecay`: Learning rate scheduler

### 3. `training_evaluation_univ_transformer.py`
- **Purpose**: Model training and evaluation
- **Contains**:
  - `train_given_model_and_data()`: Main training function
  - `evaluate_model_sliding_window()`: Sliding window evaluation
  - `evaluate_model_basic()`: Basic evaluation metrics
  - `plot_evaluations()`: Metric visualization

### 4. `visualization_univ_transformer.py`
- **Purpose**: Plotting and visualization functions
- **Contains**:
  - `plt_model()`: Model results plotting
  - `plot_predictions_with_waves()`: Predictions with pandemic waves
  - `plot_example()`: Raw data visualization
  - `plot_training_history()`: Training history plots
  - `plot_residuals_analysis()`: Residuals analysis

### 5. `utils_univ_transformer.py`
- **Purpose**: Helper functions and utilities
- **Contains**:
  - `extract_model_params()`: Parse parameters from filenames
  - `load_and_evaluate_models()`: Batch model evaluation
  - `setup_gpu_memory()`: GPU configuration
  - `create_model_directories()`: Directory setup
  - `calculate_forecast_metrics()`: Comprehensive metrics

### 6. `main_training_univ_transformer.py`
- **Purpose**: Main pipeline orchestration
- **Contains**:
  - Complete training and evaluation pipeline
  - Hyperparameter sweep functionality
  - Model comparison and analysis

## 🚀 Quick Start

### Basic Usage

```python
from univariate_transformer import (
    default_config, build_model, 
    train_given_model_and_data, plt_model
)
import data_preparation

# Load configuration
config = default_config

# Prepare data
X, Y = data_preparation.prepare_data(
    config.DATA_PATH, config.TARGET_CODE, 
    config.LOOKBACK, config.FORECAST, 
    univariate=True
)

# Build model
model = build_model(
    (config.LOOKBACK, 1),
    head_size=config.HEAD_SIZE,
    num_heads=config.NUM_HEADS,
    ff_dim=config.FF_DIM,
    num_transformer_blocks=config.NUM_TRANSFORMER_BLOCKS,
    mlp_units=[config.MLP_UNITS],
    dropout=config.DROPOUT,
    n_pred=config.FORECAST
)

# Train model
model_name = config.get_model_name()
train_given_model_and_data(model, X, Y, model_name=model_name)
```

### Custom Configuration

```python
from univariate_transformer import create_config, build_model

# Create custom configuration
config = create_config(
    lookback=14, 
    forecast=7, 
    learning_rate=0.0001,
    epochs=50
)

config.print_config()  # Display configuration
```

### Run Complete Pipeline

```python
# Run the complete training and evaluation pipeline
python main_training_univ_transformer.py
```

## 🔧 Key Features

### 1. **Modular Design**
- Clean separation of concerns
- Easy to maintain and extend
- Reusable components

### 2. **Comprehensive Configuration**
- Centralized parameter management
- Easy hyperparameter tuning
- Standardized model naming

### 3. **Robust Training Pipeline**
- GPU memory management
- Early stopping and scheduling
- Comprehensive logging

### 4. **Rich Visualization**
- Model performance plots
- Pandemic wave overlays
- Training history visualization
- Residuals analysis

### 5. **Utility Functions**
- Model parameter extraction from filenames
- Batch model evaluation
- Comprehensive metrics calculation

## 📊 Model Architecture

The transformer model includes:
- **Multi-head attention** with configurable heads and dimensions
- **Feed-forward networks** with residual connections
- **Layer normalization** for stable training
- **Dropout regularization** to prevent overfitting
- **Global average pooling** for sequence reduction
- **MLP head** for final predictions

## 🎯 Use Cases

1. **Single Model Training**: Train individual models with specific parameters
2. **Hyperparameter Sweeps**: Systematic exploration of parameter combinations  
3. **Model Comparison**: Evaluate and compare multiple trained models
4. **Production Inference**: Load and use trained models for predictions
5. **Research & Development**: Experiment with new architectures and techniques

## 📈 Evaluation Metrics

The module calculates comprehensive metrics:
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error  
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of Determination
- **Directional Accuracy**: Trend prediction accuracy

## 🔍 Troubleshooting

### Common Issues

1. **Softmax Warning**: Occurs with lookback=1. Skip these configurations or use bypass logic.
2. **GPU Memory**: Use `setup_gpu_memory()` to configure memory growth.
3. **File Paths**: Ensure data paths are correct and accessible.
4. **Dependencies**: Install required packages: `tensorflow`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.

### Best Practices

1. **Always use absolute paths** for data files
2. **Configure GPU memory** before training
3. **Use early stopping** to prevent overfitting  
4. **Save models** with descriptive names
5. **Validate data shapes** before training

## 🤝 Contributing

When adding new functionality:
1. Follow the existing naming convention (`*_univ_transformer.py`)
2. Add comprehensive docstrings
3. Update the `__init__.py` imports
4. Add examples to this README
5. Test with different parameter combinations

## 📄 License

This module is part of the TRANSFORMERS_PREDAP project.