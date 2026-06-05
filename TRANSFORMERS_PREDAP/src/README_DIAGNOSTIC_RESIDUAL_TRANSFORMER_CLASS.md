# Diagnostic Residual Transformer - Class-Based Implementation

This document describes the class-based implementation of the Diagnostic Residual Transformer, which provides a clean, reusable, and object-oriented approach to training and evaluating residual correction models.

## Overview

The Diagnostic Residual Transformer learns to correct base model predictions by training on residual errors using diagnostic covariates. This class-based implementation provides:

- **Clean Configuration Management**: Centralized configuration using dataclasses
- **Modular Pipeline**: Step-by-step execution with clear phases
- **Reusable Components**: Easy to extend and customize for different use cases
- **Comprehensive Evaluation**: Built-in visualization and performance analysis
- **Error Handling**: Robust error handling and validation

## Key Components

### 1. Configuration Classes

#### `DefaultDiagnosticConfig`
Base configuration with sensible defaults for diagnostic residual transformers:

```python
from config.default_diagnostic_config import DefaultDiagnosticConfig

config = DefaultDiagnosticConfig(
    code="T14",
    forecast=7,
    lookback=14
)
```

#### `DiagnosticResidualTransformerConfig`
Extended configuration class that inherits from `DefaultDiagnosticConfig`:

```python
from main_train_diagnostic_residual_transformer_class import DiagnosticResidualTransformerConfig

config = DiagnosticResidualTransformerConfig(
    code="J00",
    forecast=14,
    lookback=30,
    head_size=64,
    num_heads=8,
    ff_dim=128,
    dropout=0.2,
    learning_rate=0.001,
    epochs=100
)
```

### 2. Pipeline Class

#### `DiagnosticResidualTransformerPipeline`
Main class that orchestrates the entire training and evaluation process:

```python
from main_train_diagnostic_residual_transformer_class import (
    DiagnosticResidualTransformerConfig, 
    DiagnosticResidualTransformerPipeline
)

# Create configuration
config = DiagnosticResidualTransformerConfig(code="T14")

# Create pipeline
pipeline = DiagnosticResidualTransformerPipeline(config)

# Run complete pipeline
results = pipeline.run_complete_pipeline()
```

## Configuration Parameters

### Core Model Parameters
- `lookback`: Historical window size (default: 14)
- `forecast`: Prediction horizon (default: 7)  
- `code`: Target diagnostic code (e.g., "J00", "T14")

### Model Architecture
- `head_size`: Transformer attention head size (default: 32)
- `num_heads`: Number of attention heads (default: 4)
- `ff_dim`: Feed-forward dimension (default: 64)
- `dropout`: Dropout rate (default: 0.3)
- `activation_function`: Activation function (default: 'tanh')

### Training Parameters
- `learning_rate`: Learning rate (default: 0.0005)
- `epochs`: Training epochs (default: 150)
- `batch_size`: Batch size (default: 64)
- `early_stop_patience`: Early stopping patience (default: 15)

### Diagnostic-Specific Parameters
- `diagnostic_covariates_list`: List of diagnostic codes to use as covariates
- `diagnostic_covariates_path`: Path to diagnostic covariates configuration file
- `base_model_dir`: Directory containing base transformer models

### Evaluation Settings
- `plot_stepwise_errors`: Generate stepwise error plots (default: True)
- `plot_residuals_analysis`: Generate residual analysis plots (default: True)  
- `plot_pandemic_waves`: Generate pandemic wave analysis plots (default: True)
- `save_performance_results`: Save metrics to JSON (default: True)

## Usage Examples

### Basic Usage

```python
from main_train_diagnostic_residual_transformer_class import (
    DiagnosticResidualTransformerConfig, 
    DiagnosticResidualTransformerPipeline
)

# Simple usage with defaults
config = DiagnosticResidualTransformerConfig(
    code="T14",
    forecast=7,
    lookback=14
)

pipeline = DiagnosticResidualTransformerPipeline(config)
results = pipeline.run_complete_pipeline()

predictions_train_corrected, predictions_test_corrected, residual_model, residual_model_name, corrected_mae, corrected_mse, corrected_rmse = results
```

### Custom Configuration

```python
# Custom configuration for specific requirements
config = DiagnosticResidualTransformerConfig(
    code="J00",
    forecast=14,
    lookback=30,
    
    # Model architecture
    head_size=64,
    num_heads=8,
    ff_dim=128,
    dropout=0.2,
    activation_function='gelu',
    
    # Training parameters
    learning_rate=0.001,
    epochs=100,
    batch_size=32,
    
    # Custom diagnostic covariates
    diagnostic_covariates_list=['M06', 'G45', 'E78', 'I48', 'I25', 'I10']
)

pipeline = DiagnosticResidualTransformerPipeline(config)
results = pipeline.run_complete_pipeline()
```

### Step-by-Step Execution

```python
# Run pipeline step by step for more control
config = DiagnosticResidualTransformerConfig(code="T14")
pipeline = DiagnosticResidualTransformerPipeline(config)

# Step 1: Setup
pipeline.setup_environment()

# Step 2: Prepare base model data and compute residuals  
pipeline.prepare_base_model_data()

# Step 3: Prepare diagnostic covariate data
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

2. **Phase 2**: Prepare covariate data for residual model  
   - Load and filter diagnostic covariate data
   - Generate rolling sequences for training and testing

3. **Phase 3**: Train residual correction model
   - Build hybrid LSTM + Transformer model for residual learning
   - Train model using diagnostic covariates to predict residuals

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

## File Structure

```
src/
├── config/
│   ├── __init__.py
│   ├── base_transformer_config.py
│   └── default_diagnostic_config.py
├── main_train_diagnostic_residual_transformer_class.py
├── example_diagnostic_residual_transformer_usage.py
└── README_DIAGNOSTIC_RESIDUAL_TRANSFORMER_CLASS.md
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

The class-based implementation provides the same functionality as the original functional `main_train_diagnostic_residual_transformer()` function, but with better structure:

```python
# Old functional approach
predictions_train_corrected, predictions_test_corrected, residual_model, residual_model_name, corrected_mae, corrected_mse, corrected_rmse = main_train_diagnostic_residual_transformer(
    forecast=7, lookback=14, code="T14", activation_function='gelu', 
    covid_token=False, diagnostic_covariates_path='../data/diagnostic_covariates_config.txt'
    # ... many more parameters
)

# New class-based approach  
config = DiagnosticResidualTransformerConfig(forecast=7, lookback=14, code="T14")
pipeline = DiagnosticResidualTransformerPipeline(config)
results = pipeline.run_complete_pipeline()
```

## Requirements

- TensorFlow/Keras for deep learning models
- NumPy and Pandas for data manipulation
- Scikit-learn for evaluation metrics
- All modules from the original functional implementation

## Example Usage Script

See `example_diagnostic_residual_transformer_usage.py` for comprehensive examples demonstrating:
- Basic usage with default parameters
- Custom configuration for different model settings
- Multiple diagnostic code comparisons
- Step-by-step pipeline execution

Run examples with:
```bash
python example_diagnostic_residual_transformer_usage.py
```