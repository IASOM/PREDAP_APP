# LMLR Module API Reference

The Lagged Multiple Linear Regression (LMLR) module provides tools for feature selection, correlation analysis, and statistical modeling for time series data.

## Overview

The LMLR module consists of several submodules:

- **`correlation_lmlr`**: Correlation analysis and multicollinearity detection
- **`modeling_lmlr`**: Model training and selection utilities  
- **`metrics_lmlr`**: Performance evaluation and statistical testing
- **`preprocessing_lmlr`**: Data preprocessing and smoothing functions
- **`visualization_lmlr`**: Plotting and visualization tools

## Correlation Analysis

The correlation analysis tools help identify relationships between variables and detect multicollinearity:

- **get_top_correlations_blog**: Identifies the strongest correlations between predictors and target variables
- **compute_vif**: Calculates Variance Inflation Factor to detect multicollinearity  
- **filter_VIF**: Removes variables with high VIF values to reduce multicollinearity

## Model Training and Selection

Comprehensive model training and selection utilities:

- **models_training**: Trains multiple regression models (Linear, Ridge, Lasso, etc.)
- **select_best_models**: Identifies best performing models based on validation metrics
- **select_best_absolute_model**: Selects the single best model across all criteria

## Performance Metrics

Statistical evaluation tools for model assessment:

- **metrics_calculation**: Computes comprehensive performance metrics (MAE, MSE, RMSE, R²)
- **evaluation_metrics_MAPE**: Calculates Mean Absolute Percentage Error
- **evaluation_metrics_RMSE**: Computes Root Mean Square Error with confidence intervals
- **evaluation_metrics_Ftest**: Performs F-test for model significance

## Data Preprocessing

Data preparation and smoothing functions:

- **smoother**: Applies smoothing techniques to reduce noise in time series data
- Supports various smoothing methods (moving average, exponential smoothing)

## Visualization Tools

Plotting and visualization utilities:

- **plot_example**: Creates example plots for model demonstration
- **ploter**: General plotting function for predictions vs actual values
- **residual_plots**: Generates residual analysis plots for model diagnostics

::: src.lmlr.visualization_lmlr.plot_models

::: src.lmlr.visualization_lmlr.plot_metrics

## Usage Examples

### Basic Correlation Analysis

```python
from src.lmlr import get_top_correlations_blog, compute_vif, filter_VIF

# Find highly correlated variables
correlations = get_top_correlations_blog(df, threshold=0.90)
print(f"Found {len(correlations)} highly correlated pairs")

# Calculate VIF for multicollinearity
correlated_vars = list(set(correlations.index.get_level_values(0)))
vif = compute_vif(correlated_vars, df)

# Filter variables based on VIF threshold
filtered_vars = filter_VIF(vif, df, iterations_max=400, VIF_threshold=20.0)
```

### Model Training Workflow

```python
from src.lmlr import models_training, evaluation_metrics_MAPE

# Train models with incremental feature addition
results = models_training(
    df=scaled_data,
    code='target_variable',
    corr=correlation_series,
    max_iters=60,
    plt_models=True,
    plt_metrics=True
)

# Get best model
best_model = results[results["BEST_MODEL"] == "YES"]

# Evaluate performance
evaluation_metrics_MAPE(results)
```

### Data Preprocessing

```python
from src.lmlr import smoother, plot_example

# Apply smoothing to reduce noise
smoothed_data = smoother(df, window_size=14)

# Visualize smoothed data
plot_example(smoothed_data, "Smoothed Time Series (10 examples)")
```

## Configuration Parameters

### Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.90 | Correlation threshold for variable filtering |
| `VIF_threshold` | float | 20.0 | VIF threshold for multicollinearity filtering |
| `max_iters` | int | 60 | Maximum iterations for model training |
| `window_size` | int | 14 | Window size for smoothing operations |
| `pval` | float | 0.1 | P-value threshold for model selection |

### Model Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `plt_models` | bool | True | Whether to plot model comparisons |
| `plt_metrics` | bool | True | Whether to plot performance metrics |
| `plt_selected_models` | bool | True | Whether to plot selected models |
| `plt_best_model` | bool | True | Whether to plot the best model |

## Return Types

### Models Training Output

The `models_training` function returns a DataFrame with the following columns:

- `number_of_variables`: Number of variables in the model
- `F1`: F-statistic for nested model comparison
- `pval1`: P-value for F1 statistic
- `F2`: Alternative F-statistic formulation
- `pval2`: P-value for F2 statistic
- `MAPE_train`: Mean Absolute Percentage Error on training data
- `MAPE_test`: Mean Absolute Percentage Error on test data
- `RMSE_train`: Root Mean Square Error on training data
- `RMSE_test`: Root Mean Square Error on test data
- `predictors`: Comma-separated list of predictor variables
- `BEST_MODEL`: "YES" for the selected best model, "NO" otherwise

### VIF Output

The `compute_vif` function returns a DataFrame with:

- `Variable`: Variable name
- `VIF`: Variance Inflation Factor value

## Error Handling

The LMLR module includes robust error handling for common issues:

- **Insufficient Data**: Minimum sample size requirements for reliable statistics
- **Multicollinearity**: Automatic detection and handling of highly correlated variables
- **Convergence Issues**: Iterative algorithms include maximum iteration limits
- **Missing Values**: Appropriate handling of NaN values in time series data

## Performance Considerations

- **Memory Usage**: Large datasets may require chunking for VIF calculations
- **Computation Time**: Model training time scales with number of features and iterations
- **Statistical Assumptions**: Ensure data meets assumptions for linear regression

## Related Functions

See also:

- [Granger Causality Module](gcausal.md) for causal analysis
- [Deep Learning Module](dl.md) for neural network forecasting
- [Utilities](utilities.md) for helper functions