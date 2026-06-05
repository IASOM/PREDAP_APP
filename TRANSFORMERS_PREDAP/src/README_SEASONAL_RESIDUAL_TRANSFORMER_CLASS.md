# Seasonal Residual Transformer - Class-Based Implementation

This document describes the class-based implementation of the Seasonal Residual Transformer, which provides a clean, reusable, and object-oriented approach to training and evaluating residual correction models using seasonal covariates.

## Overview

The Seasonal Residual Transformer learns to correct base model predictions by training on residual errors using seasonal time series features. This class-based implementation provides:

- **Clean Configuration Management**: Centralized configuration using dataclasses
- **Modular Pipeline**: Step-by-step execution with clear phases
- **Seasonal Feature Engineering**: Automated creation of seasonal covariates
- **Reusable Components**: Easy to extend and customize for different use cases
- **Comprehensive Evaluation**: Built-in visualization and performance analysis
- **Error Handling**: Robust error handling and validation

## Key Components

### 1. Configuration Classes

#### `DefaultSeasonalConfig`
Base configuration with sensible defaults for seasonal residual transformers:

```python
from config.default_seasonal_config import DefaultSeasonalConfig

config = DefaultSeasonalConfig(
    code="T14",
    forecast=7,
    lookback=14
)
```

#### `SeasonalResidualTransformerConfig`
Extended configuration class that inherits from `DefaultSeasonalConfig`:

```python
from main_train_seasonal_residual_transformer_class import SeasonalResidualTransformerConfig

config = SeasonalResidualTransformerConfig(
    code="J00",
    forecast=14,
    lookback=30,
    head_size=4,
    num_heads=4,
    ff_dim=16,
    dropout=0.1,
    learning_rate=0.0005,
    epochs=100,
    categorical_vars=["Day_of_Week", "Month", "Season", "Holiday"]
)
```

### 2. Pipeline Class

#### `SeasonalResidualTransformerPipeline`
Main class that orchestrates the entire training and evaluation process:

```python
from main_train_seasonal_residual_transformer_class import (
    SeasonalResidualTransformerConfig, 
    SeasonalResidualTransformerPipeline
)

# Create configuration
config = SeasonalResidualTransformerConfig(code="T14")

# Create pipeline
pipeline = SeasonalResidualTransformerPipeline(config)

# Run complete pipeline
results = pipeline.run_complete_pipeline()
```

## Configuration Parameters

### Core Model Parameters
- `lookback`: Historical window size (default: 14)
- `forecast`: Prediction horizon (default: 7)  
- `code`: Target diagnostic code (e.g., "J00", "T14")

### Model Architecture
- `head_size`: Transformer attention head size (default: 2)
- `num_heads`: Number of attention heads (default: 2)
- `ff_dim`: Feed-forward dimension (default: 8)
- `mlp_units`: MLP units (default: 64)
- `dropout`: Dropout rate (default: 0.2)
- `activation_function`: Activation function (default: 'tanh')

### Training Parameters
- `learning_rate`: Learning rate (default: 0.001)
- `epochs`: Training epochs (default: 100)
- `batch_size`: Batch size (default: 32)
- `early_stop_patience`: Early stopping patience (default: 10)

### Seasonal-Specific Parameters
- `categorical_vars`: List of seasonal features to use as covariates
  - Available options: `["Day_of_Week", "Month", "Season", "Holiday", "School_Vacation"]`
  - Default: All available features

### Evaluation Settings
- `plot_stepwise_errors`: Generate stepwise error plots (default: True)
- `plot_residuals_analysis`: Generate residual analysis plots (default: True)  
- `plot_pandemic_waves`: Generate pandemic wave analysis plots (default: True)
- `save_performance_results`: Save metrics to JSON (default: True)

## Seasonal Features

The pipeline automatically generates the following seasonal features from timestamp data:

1. **Day_of_Week**: Day of the week (0-6, Monday=0)
2. **Month**: Month of the year (1-12)
3. **Season**: Meteorological season (Spring=0, Summer=1, Autumn=2, Winter=3)
4. **Holiday**: Binary indicator for public holidays
5. **School_Vacation**: Binary indicator for school vacation periods

These features are engineered using the `data_preparation.prepare_time_series_features()` function and can be selectively included via the `categorical_vars` parameter.

## Usage Examples

### Basic Usage

```python
from main_train_seasonal_residual_transformer_class import (
    SeasonalResidualTransformerConfig, 
    SeasonalResidualTransformerPipeline
)

# Simple usage with defaults
config = SeasonalResidualTransformerConfig(
    code="T14",
    forecast=7,
    lookback=14
)

pipeline = SeasonalResidualTransformerPipeline(config)
results = pipeline.run_complete_pipeline()

predictions_train_corrected, predictions_test_corrected, residual_model, residual_model_name, corrected_mae, corrected_mse, corrected_rmse = results
```

### Custom Seasonal Features

```python
# Use only specific seasonal features
config = SeasonalResidualTransformerConfig(
    code="J00",
    forecast=14,
    lookback=30,
    
    # Select specific seasonal features
    categorical_vars=["Day_of_Week", "Month", "Season"],
    
    # Model architecture
    head_size=4,
    num_heads=4,
    ff_dim=16,
    dropout=0.1,
    
    # Training parameters
    learning_rate=0.0005,
    epochs=50,
    batch_size=64
)

pipeline = SeasonalResidualTransformerPipeline(config)
results = pipeline.run_complete_pipeline()
```

### Step-by-Step Execution

```python
# Run pipeline step by step for more control
config = SeasonalResidualTransformerConfig(code="T14")
pipeline = SeasonalResidualTransformerPipeline(config)

# Step 1: Setup
pipeline.setup_environment()

# Step 2: Prepare base model data and compute residuals  
pipeline.prepare_base_model_data()

# Step 3: Prepare seasonal covariate data
pipeline.prepare_covariate_data()

# Step 4: Train residual model
pipeline.train_residual_model()

# Step 5: Evaluate residual model
pipeline.evaluate_residual_model()

# Step 6: Generate visualizations
predictions_to_plot, corrected_to_plot, Y_test_to_plot = pipeline.generate_visualizations()

# Step 7: Calculate metrics
corrected_mae, corrected_mse, corrected_rmse = pipeline.calculate_performance_metrics(
    predictions_to_plot, corrected_to_plot, Y_test_to_plot
)
```

## Pipeline Phases

The pipeline consists of 6 main phases:

1. **Phase 1**: Load base model and compute residuals
   - Load base transformer model predictions
   - Calculate residual errors between predictions and actual values

2. **Phase 2**: Prepare seasonal covariate data for residual model  
   - Generate seasonal time series features (Day_of_Week, Month, Season, etc.)
   - Create rolling sequences for training and testing

3. **Phase 3**: Train residual correction model
   - Build hybrid LSTM + Transformer model for residual learning
   - Train model using seasonal covariates to predict residuals

4. **Phase 4**: Evaluate residual correction model
   - Apply trained residual model to test data
   - Generate corrected predictions by adding predicted residuals

5. **Phase 5**: Visualization and analysis
   - Generate comprehensive plots and visualizations
   - Analyze performance across different time periods and pandemic waves

6. **Phase 6**: Performance summary
   - Calculate evaluation metrics (MAE, MSE, RMSE)
   - Compare original vs corrected model performance
   - Save results to JSON files

## Output and Results

The pipeline returns:
- `predictions_train_corrected`: Corrected training predictions
- `predictions_test_corrected`: Corrected test predictions  
- `residual_model`: Trained residual correction model
- `residual_model_name`: Name of the saved residual model
- `corrected_mae`: Mean Absolute Error of corrected predictions
- `corrected_mse`: Mean Squared Error of corrected predictions
- `corrected_rmse`: Root Mean Squared Error of corrected predictions

## Performance Analysis

The pipeline automatically generates:
- Stepwise error comparison plots
- Residual analysis visualizations
- Predictions vs actual values with pandemic wave annotations
- Error significance analysis during pandemic periods
- Performance improvement metrics and percentages

## Seasonal Feature Engineering

### Automatic Feature Generation

The pipeline automatically processes timestamp data to create seasonal features:

```python
# Generated features include:
categorical_vars = [
    "Day_of_Week",      # 0-6 (Monday=0)
    "Month",            # 1-12 
    "Season",           # 0-3 (Spring=0, Summer=1, Autumn=2, Winter=3)
    "Holiday",          # 0/1 Binary
    "School_Vacation"   # 0/1 Binary
]
```

### Custom Feature Selection

You can select which seasonal features to use:

```python
# Use only day and month patterns
config = SeasonalResidualTransformerConfig(
    categorical_vars=["Day_of_Week", "Month"]
)

# Use comprehensive seasonal information
config = SeasonalResidualTransformerConfig(
    categorical_vars=["Day_of_Week", "Month", "Season", "Holiday", "School_Vacation"]
)
```

## File Structure

```
src/
├── config/
│   ├── __init__.py
│   ├── base_transformer_config.py
│   └── default_seasonal_config.py
├── main_train_seasonal_residual_transformer_class.py
├── example_seasonal_residual_transformer_usage.py
└── README_SEASONAL_RESIDUAL_TRANSFORMER_CLASS.md
```

## Comparison with Functional Implementation

### Advantages of Class-Based Approach:

1. **Better Organization**: Clear separation of configuration, data preparation, training, and evaluation
2. **Reusability**: Easy to reuse components for different experiments
3. **Maintainability**: Easier to modify and extend functionality
4. **Error Handling**: Robust error handling and parameter validation
5. **Configuration Management**: Centralized configuration with inheritance
6. **Modularity**: Can run individual phases or complete pipeline
7. **Results Tracking**: Built-in results summary and metrics tracking

### Migration from Functional Implementation:

The class-based implementation provides the same functionality as the original functional `main_train_seasonal_residual_transformer()` function, but with better structure:

```python
# Old functional approach
predictions_train_corrected, predictions_test_corrected, residual_model, residual_model_name, corrected_mae, corrected_mse, corrected_rmse = main_train_seasonal_residual_transformer(
    lookback=14, forecast=7, code="T14", activation_function='tanh', 
    covid_token=False, head_size=2, num_heads=2, ff_dim=8
    # ... many more parameters
)

# New class-based approach  
config = SeasonalResidualTransformerConfig(lookback=14, forecast=7, code="T14")
pipeline = SeasonalResidualTransformerPipeline(config)
results = pipeline.run_complete_pipeline()
```

## Requirements

- TensorFlow/Keras for deep learning models
- NumPy and Pandas for data manipulation
- Scikit-learn for evaluation metrics
- All modules from the original functional implementation

## Example Usage Script

See `example_seasonal_residual_transformer_usage.py` for comprehensive examples demonstrating:
- Basic usage with default parameters
- Custom configuration for different model settings
- Seasonal features comparison across different combinations
- Multiple diagnostic code comparisons
- Step-by-step pipeline execution

Run examples with:
```bash
python example_seasonal_residual_transformer_usage.py
```

## Key Differences from Diagnostic Residual Transformer

1. **Feature Type**: Uses seasonal time series features instead of diagnostic covariates
2. **Feature Engineering**: Automatic generation of temporal features from timestamps
3. **Model Architecture**: Optimized for seasonal patterns with smaller architectures
4. **Configuration**: Specialized for temporal feature selection and engineering
5. **Use Case**: Better suited for capturing temporal/seasonal patterns in residuals