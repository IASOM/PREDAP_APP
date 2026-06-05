# Quick Start Guide

Get up and running with CCLR-PREDAP in minutes! This guide demonstrates the basic usage with synthetic data.

## 1. Basic Setup

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import CCLR-PREDAP modules
from src.lmlr import (
    get_top_correlations_blog, compute_vif, filter_VIF,
    models_training, evaluation_metrics_MAPE
)
from src.gcausal import (
    kpss_test, stationate, granger_causation_matrix,
    select_causal_features, min_max_scale
)
from src.dl import (
    create_model_lstm, split_sequence, fit_model,
    prediction, inverse_transform, evaluate_forecast
)
```

## 2. Load Sample Data

```python
# Load the synthetic dataset
data_path = 'data/synthetic_timeseries.csv'
df = pd.read_csv(data_path, index_col=0)

# Set datetime index
df.index = pd.date_range(start="2010-01-01", periods=len(df), freq="D")

# Basic preprocessing
df = df.clip(lower=0)  # Remove negative values
df_scaled = min_max_scale(df)  # Normalize data

print(f"Dataset shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
```

## 3. Phase 1: Feature Selection (LMLR)

### Correlation Analysis and VIF Filtering

```python
# Configuration
TARGET_VARIABLE = 'timeseries_350'  # Choose your target
CORRELATION_THRESHOLD = 0.90
VIF_THRESHOLD = 20.0

# Step 1: Find highly correlated variables
df_correlations = get_top_correlations_blog(
    df_scaled, 
    threshold=CORRELATION_THRESHOLD
)
print(f"Found {len(df_correlations)} highly correlated pairs")

# Step 2: VIF filtering to reduce multicollinearity
correlated_vars = list(set(
    df_correlations.index.get_level_values(0).tolist() + 
    df_correlations.index.get_level_values(1).tolist()
))

if correlated_vars:
    vif = compute_vif(correlated_vars, df_scaled)
    filtered_vars = filter_VIF(vif, df_scaled, 400, VIF_THRESHOLD)
    print(f"Reduced from {len(correlated_vars)} to {len(filtered_vars)} variables")
```

### Model Training and Selection

```python
# Step 3: Train lagged regression models
best_features = models_training(
    df_scaled, 
    TARGET_VARIABLE, 
    corr=df_scaled.corrwith(df_scaled[TARGET_VARIABLE]).sort_values(ascending=False, key=abs),
    max_iters=50
)

print("Best features selected:")
print(best_features[best_features["BEST_MODEL"] == "YES"])
```

## 4. Phase 2: Granger Causality Testing

### Stationarity Check and Transformation

```python
# Extract predictor variables from best features
predictors = best_features[best_features["BEST_MODEL"] == "YES"]["predictors"].iloc[0]
predictor_list = [p.strip() for p in predictors.split(',')]

# Prepare variables for causality testing
variables = [TARGET_VARIABLE] + predictor_list
subset_df = df[variables]

# Check stationarity
stationarity_results = kpss_test(subset_df)
print("Stationarity test results:")
print(stationarity_results)

# Make non-stationary series stationary
non_stationary = stationarity_results[stationarity_results['p-value'] < 0.05].index.tolist()
if non_stationary:
    subset_df = stationate(subset_df, non_stationary)
    print(f"Applied differencing to {len(non_stationary)} variables")
```

### Granger Causality Analysis

```python
# Test for Granger causality
causality_matrix = granger_causation_matrix(
    subset_df, 
    subset_df.columns, 
    p=7  # 7-day lag
)

print("Granger Causality Matrix:")
print(causality_matrix)

# Select causal features
causal_features = select_causal_features(
    causality_matrix, 
    target_variable=TARGET_VARIABLE,
    significance_level=0.05
)

print(f"Causal features: {causal_features}")
```

## 5. Phase 3: Deep Learning Forecasting

### Data Preparation

```python
from sklearn.preprocessing import MinMaxScaler

# Prepare final dataset with causal features
final_features = causal_features + [TARGET_VARIABLE]
final_df = df[final_features]

# Train-test split
train_size = int(len(final_df) * 0.8)
train_data = final_df.iloc[:train_size]
test_data = final_df.iloc[train_size:]

# Scale data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

# Create sequences for LSTM
LOOK_BACK = 30
FORECAST_HORIZON = 7

X_train, y_train = split_sequence(train_scaled, LOOK_BACK, FORECAST_HORIZON)
X_test, y_test = split_sequence(test_scaled, LOOK_BACK, FORECAST_HORIZON)

print(f"Training sequences: {X_train.shape}")
print(f"Test sequences: {X_test.shape}")
```

### Model Training

```python
# Create and train LSTM model
model = create_model_lstm(X_train)

# Train model
history = fit_model(
    model, 
    X_train, 
    y_train,
    epochs=50,
    batch_size=32,
    validation=0.2,
    patience=10
)

print("Model training completed!")
```

### Prediction and Evaluation

```python
# Make predictions
predictions = prediction(model, X_test)

# Inverse transform to original scale
y_test_orig, pred_orig = inverse_transform(y_test, predictions, scaler)

# Evaluate model performance
metrics = evaluate_forecast(y_test_orig, pred_orig)
print("Model Performance:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
```

## 6. Visualization

```python
import matplotlib.pyplot as plt

# Plot predictions vs actual values
plt.figure(figsize=(15, 6))

# Get target column index
target_idx = final_df.columns.get_loc(TARGET_VARIABLE)

plt.plot(y_test_orig[:, target_idx], label='Actual', linewidth=2)
plt.plot(pred_orig[:, target_idx], label='Predicted', linestyle='--', linewidth=2)

plt.title(f'CCLR-PREDAP Forecast Results: {TARGET_VARIABLE}')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## 7. Complete Pipeline Function

Here's a convenient wrapper function for the entire pipeline:

```python
def run_cclr_predap_pipeline(df, target_variable, config=None):
    """
    Run the complete CCLR-PREDAP pipeline
    
    Args:
        df: Input DataFrame with time series data
        target_variable: Name of the target variable to predict
        config: Configuration dictionary (optional)
    
    Returns:
        dict: Results including predictions, metrics, and selected features
    """
    
    # Default configuration
    if config is None:
        config = {
            'correlation_threshold': 0.90,
            'vif_threshold': 20.0,
            'max_iters': 50,
            'causality_lag': 7,
            'look_back': 30,
            'forecast_horizon': 7,
            'epochs': 50,
            'batch_size': 32
        }
    
    results = {}
    
    # Phase 1: LMLR
    print("Phase 1: Running LMLR feature selection...")
    # ... (implement full pipeline)
    
    # Phase 2: Granger Causality
    print("Phase 2: Testing Granger causality...")
    # ... (implement causality testing)
    
    # Phase 3: Deep Learning
    print("Phase 3: Training deep learning models...")
    # ... (implement DL training)
    
    return results

# Usage
results = run_cclr_predap_pipeline(df, 'timeseries_350')
```

## Next Steps

Now that you've completed the quick start:

1. **Explore Advanced Features**: Check out [Advanced Examples](../examples/advanced.md)
2. **Understand the Methodology**: Read the detailed [User Guide](../user-guide/overview.md)
3. **Customize Your Analysis**: Learn about [Configuration Options](configuration.md)
4. **API Documentation**: Browse the complete [API Reference](../api/lmlr.md)

## Common Next Steps

- **Hyperparameter Tuning**: Optimize model parameters for your specific data
- **Multi-Target Forecasting**: Extend to predict multiple variables simultaneously
- **Real-time Prediction**: Set up continuous forecasting pipelines
- **Model Comparison**: Evaluate different deep learning architectures

---

**Tip**: Start with the synthetic data to understand the workflow, then apply it to your own time series data!