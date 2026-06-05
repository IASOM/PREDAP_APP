# Granger Causality Module API Reference

The Granger Causality module provides tools for causal analysis, stationarity testing, and time series preprocessing for causality analysis.

## Overview

The Granger Causality module consists of several submodules:

- **`causality_gcausal`**: Granger causality testing and feature selection
- **`stationarity_gcausal`**: Stationarity testing and transformation
- **`preprocessing_gcausal`**: Data preprocessing for causality analysis
- **`model_selection_gcausal`**: VAR model selection and optimization
- **`visualization_gcausal`**: Lag plots and causality visualization

## Causality Testing

Tools for performing Granger causality analysis:

- **granger_causation_matrix**: Computes pairwise Granger causality tests between all variables
- **select_causal_features**: Identifies variables that have significant causal relationships with the target

## Stationarity Analysis

Statistical tests and transformations for time series stationarity:

- **kpss_test**: Performs KPSS test to check for stationarity
- **stationate**: Applies differencing or other transformations to achieve stationarity

## Model Selection

VAR model optimization and lag order selection:

- **select_p**: Determines optimal lag order for VAR models using information criteria
- **recommend_lag_order**: Provides lag order recommendations based on multiple criteria  
- **fit_var_model**: Fits Vector Autoregression models with selected parameters

## Data Preprocessing

Specialized preprocessing for causality analysis:

- **min_max_scale**: Applies min-max scaling to normalize variables
- **splitter**: Splits time series data while preserving temporal structure

## Visualization Tools

Visual analysis tools for causality:

- **lag_plots**: Creates lag plots to visualize temporal relationships between variables
- **causality_heatmap**: Generates heatmaps showing causal relationships strength

## Usage Examples

### Basic Granger Causality Analysis

```python
from src.gcausal import (
    kpss_test, stationate, granger_causation_matrix, 
    select_causal_features
)

# Test for stationarity
stationarity_results = kpss_test(df)
print("Non-stationary variables:")
non_stationary = stationarity_results[stationarity_results['p-value'] < 0.05].index
print(non_stationary.tolist())

# Make series stationary
if len(non_stationary) > 0:
    df_stationary = stationate(df, non_stationary.tolist())
else:
    df_stationary = df

# Test Granger causality
causality_matrix = granger_causation_matrix(
    df_stationary, 
    df_stationary.columns, 
    p=7  # 7-day lag
)

print("Granger Causality Matrix:")
print(causality_matrix)
```

### Causal Feature Selection

```python
# Select features that Granger-cause the target
causal_features = select_causal_features(
    causality_matrix,
    target_variable='your_target',
    significance_level=0.05
)

print(f"Selected causal features: {causal_features}")
```

### VAR Model Selection

```python
from src.gcausal import select_p, recommend_lag_order, fit_var_model, splitter

# Split data for model selection
train_df, test_df = splitter(df_stationary)

# Find optimal lag order
results_df, optimal_lags = select_p(train_df)
recommended_lag = recommend_lag_order(optimal_lags)

print(f"Recommended lag order: {recommended_lag}")

# Fit VAR model
var_model = fit_var_model(train_df, recommended_lag)
print("VAR model fitted successfully")
```

### Visualization

```python
from src.gcausal import lag_plots

# Create lag plots to visualize temporal patterns
variables_to_plot = ['target_var', 'predictor1', 'predictor2']
lag_plots(df[variables_to_plot])
```

## Configuration Parameters

### Stationarity Testing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `regression` | str | 'c' | Type of regression ('c' for constant, 'ct' for constant+trend) |
| `nlags` | int | None | Number of lags to use in test |

### Granger Causality Testing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `p` | int | 1 | Number of lags to use in Granger causality test |
| `test` | str | 'ssr_ftest' | Type of test to perform |
| `verbose` | bool | False | Whether to print detailed results |

### Feature Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `significance_level` | float | 0.05 | P-value threshold for causality |
| `target_variable` | str | Required | Name of target variable |

## Return Types

### KPSS Test Output

Returns a DataFrame with:

- `Variable`: Variable name
- `KPSS Statistic`: Test statistic value
- `p-value`: P-value of the test
- `Critical Values`: Dictionary of critical values
- `Stationary`: Boolean indicating stationarity

### Granger Causality Matrix

Returns a DataFrame where:
- Rows represent potential causes
- Columns represent effects
- Values are p-values (lower = stronger causality)

### VAR Model Selection

The `select_p` function returns:
- `results_df`: DataFrame with model selection criteria for different lag orders
- `optimal_lags`: Dictionary with optimal lag for each criterion

## Statistical Interpretation

### KPSS Test

- **Null Hypothesis**: Series is stationary
- **p-value < 0.05**: Reject null → Series is non-stationary
- **p-value ≥ 0.05**: Fail to reject null → Series is stationary

### Granger Causality

- **Null Hypothesis**: Variable X does not Granger-cause variable Y
- **p-value < 0.05**: Reject null → X Granger-causes Y
- **p-value ≥ 0.05**: Fail to reject null → No Granger causality

## Best Practices

### Data Preparation

1. **Check for Stationarity**: Always test and ensure stationarity before causality testing
2. **Handle Missing Values**: Remove or interpolate missing values appropriately
3. **Scale Data**: Consider scaling if variables have very different magnitudes

### Model Selection

1. **Lag Order Selection**: Use information criteria (AIC, BIC) for optimal lag selection
2. **Sample Size**: Ensure sufficient observations for reliable causality testing
3. **Multiple Testing**: Consider correction for multiple comparisons

### Interpretation

1. **Causality vs Correlation**: Granger causality indicates predictive causality, not true causation
2. **Bidirectional Causality**: Check both directions (X→Y and Y→X)
3. **Lag Selection**: Results may be sensitive to lag order choice

## Error Handling

Common issues and solutions:

- **Non-Stationary Data**: Automatically detected and handled with differencing
- **Insufficient Data**: Minimum sample size requirements enforced
- **Multicollinearity**: May affect VAR model estimation
- **Missing Values**: Handled through appropriate interpolation or removal

## Performance Considerations

- **Computational Complexity**: Scales with number of variables and lag order
- **Memory Usage**: Large VAR models may require significant memory
- **Numerical Stability**: May encounter issues with highly collinear data

## Related Functions

See also:

- [LMLR Module](lmlr.md) for feature selection
- [Deep Learning Module](dl.md) for forecasting with selected features
- [Utilities](utilities.md) for data preprocessing helpers