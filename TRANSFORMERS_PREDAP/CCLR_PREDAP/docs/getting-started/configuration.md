# Configuration Guide

This guide explains how to configure CCLR-PREDAP for your specific use case and dataset.

## Configuration Overview

CCLR-PREDAP can be configured through:

1. **Configuration Files**: YAML/JSON files for reproducible setups
2. **Environment Variables**: System-level configuration
3. **Function Parameters**: Direct parameter passing
4. **Config Classes**: Object-oriented configuration management

## Configuration File Structure

### Basic Configuration (config.yaml)

```yaml
# CCLR-PREDAP Configuration File
project:
  name: "Healthcare Demand Forecasting"
  description: "COVID-19 impact analysis"
  output_dir: "results/"

data:
  input_file: "data/synthetic_timeseries.csv"
  target_variable: "timeseries_350"
  date_column: null  # Use index if null
  start_date: "2010-01-01"
  frequency: "D"  # Daily frequency

# Phase 1: LMLR Configuration
lmlr:
  correlation_threshold: 0.90
  vif_threshold: 20.0
  max_iterations: 400
  max_iters_model: 60
  max_lag: 30
  window_size: 14  # For smoothing

# Phase 2: Granger Causality Configuration
gcausal:
  significance_level: 0.05
  max_lag_order: 30
  test_type: "ssr_ftest"
  stationarity_method: "kpss"
  differencing_max_order: 2

# Phase 3: Deep Learning Configuration
deep_learning:
  models:
    - "gru"
    - "lstm" 
    - "bilstm"
    - "encoder_decoder"
    - "cnn_lstm"
  
  training:
    epochs: 100
    batch_size: 32
    validation_split: 0.2
    patience: 10
    learning_rate: 0.001
  
  architecture:
    look_back: 30
    forecast_horizon: 7
    units: 50
    dropout: 0.2
    
  optimization:
    optimizer: "adam"
    loss: "mse"
    metrics: ["mae"]

# Visualization Configuration
visualization:
  enabled: true
  save_plots: true
  plot_format: "png"
  dpi: 300
  figsize: [12, 8]

# Logging Configuration
logging:
  level: "INFO"
  file: "logs/cclr_predap.log"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## Loading Configuration

### From YAML File

```python
import yaml
from pathlib import Path

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Usage
config = load_config('config.yaml')
```

### Configuration Class

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class LMLRConfig:
    correlation_threshold: float = 0.90
    vif_threshold: float = 20.0
    max_iterations: int = 400
    max_iters_model: int = 60
    max_lag: int = 30
    window_size: int = 14

@dataclass
class GCausalConfig:
    significance_level: float = 0.05
    max_lag_order: int = 30
    test_type: str = "ssr_ftest"
    stationarity_method: str = "kpss"
    differencing_max_order: int = 2

@dataclass
class DeepLearningConfig:
    models: List[str] = None
    epochs: int = 100
    batch_size: int = 32
    validation_split: float = 0.2
    patience: int = 10
    learning_rate: float = 0.001
    look_back: int = 30
    forecast_horizon: int = 7
    units: int = 50
    dropout: float = 0.2

@dataclass
class CCLRConfig:
    lmlr: LMLRConfig = None
    gcausal: GCausalConfig = None
    deep_learning: DeepLearningConfig = None
    
    def __post_init__(self):
        if self.lmlr is None:
            self.lmlr = LMLRConfig()
        if self.gcausal is None:
            self.gcausal = GCausalConfig()
        if self.deep_learning is None:
            self.deep_learning = DeepLearningConfig()
            self.deep_learning.models = ["lstm", "gru", "bilstm"]

# Usage
config = CCLRConfig()
```

## Environment Variables

Set environment variables for system-level configuration:

```bash
# Data configuration
export CCLR_DATA_PATH="/path/to/data"
export CCLR_OUTPUT_PATH="/path/to/output"
export CCLR_LOG_LEVEL="INFO"

# Model configuration
export CCLR_MAX_EPOCHS="200"
export CCLR_BATCH_SIZE="64"

# Hardware configuration
export CCLR_USE_GPU="true"
export CCLR_GPU_MEMORY_LIMIT="4096"
```

```python
import os

# Reading environment variables
data_path = os.getenv('CCLR_DATA_PATH', 'data/default.csv')
max_epochs = int(os.getenv('CCLR_MAX_EPOCHS', '100'))
use_gpu = os.getenv('CCLR_USE_GPU', 'false').lower() == 'true'
```

## Use Case Specific Configurations

### Healthcare Demand Forecasting

```yaml
# Healthcare-specific configuration
project:
  domain: "healthcare"
  use_case: "demand_forecasting"

data:
  seasonal_patterns: true
  holiday_effects: true
  pandemic_period: ["2020-03-01", "2021-12-31"]

lmlr:
  correlation_threshold: 0.85  # Lower for healthcare data
  max_lag: 14  # 2-week lag for medical patterns

gcausal:
  significance_level: 0.01  # Stricter for medical decisions
  max_lag_order: 14

deep_learning:
  forecast_horizon: 14  # 2-week forecast
  look_back: 60  # Consider 2-month history
  
preprocessing:
  handle_zeros: "interpolate"  # Handle zero patient counts
  outlier_detection: true
  seasonal_decomposition: true
```

### Financial Time Series

```yaml
# Financial markets configuration
project:
  domain: "finance"
  use_case: "price_forecasting" 

data:
  market_hours_only: true
  handle_weekends: "remove"
  volatility_adjustment: true

lmlr:
  correlation_threshold: 0.95  # High correlation common in finance
  max_lag: 5  # Short lags for high-frequency data

deep_learning:
  forecast_horizon: 1  # Next-day prediction
  look_back: 20  # 20-day moving window
  models: ["lstm", "gru"]  # Fast models for real-time
  
risk_management:
  confidence_intervals: true
  var_calculation: true
  stress_testing: true
```

### IoT Sensor Data

```yaml
# IoT sensor configuration
project:
  domain: "iot"
  use_case: "sensor_forecasting"

data:
  sampling_frequency: "5min"
  missing_data_threshold: 0.1
  sensor_drift_correction: true

lmlr:
  correlation_threshold: 0.90
  max_lag: 288  # 24 hours of 5-minute data

deep_learning:
  forecast_horizon: 12  # 1-hour ahead (12 * 5min)
  look_back: 288  # 24-hour lookback
  models: ["cnn_lstm", "encoder_decoder"]  # Complex patterns
  
preprocessing:
  noise_filtering: true
  anomaly_detection: true
  calibration_adjustment: true
```

## Advanced Configuration Options

### Model Ensemble Configuration

```yaml
ensemble:
  enabled: true
  method: "weighted_average"  # Options: simple, weighted_average, stacking
  weights: "performance_based"  # Options: equal, performance_based, custom
  
  stacking:
    meta_learner: "linear_regression"
    cv_folds: 5
    
  performance_weights:
    metric: "rmse"  # Metric to base weights on
    inverse: true   # Lower is better
```

### Hyperparameter Optimization

```yaml
hyperparameter_optimization:
  enabled: true
  method: "optuna"  # Options: optuna, grid_search, random_search
  n_trials: 100
  
  search_space:
    deep_learning:
      epochs: [50, 200]
      batch_size: [16, 32, 64, 128]
      learning_rate: [0.0001, 0.01]
      units: [32, 50, 100, 200]
      dropout: [0.1, 0.5]
    
    lmlr:
      correlation_threshold: [0.80, 0.95]
      vif_threshold: [10.0, 30.0]
```

### Distributed Computing

```yaml
distributed:
  enabled: false
  backend: "dask"  # Options: dask, ray, multiprocessing
  
  dask:
    scheduler: "threads"  # Options: threads, processes, distributed
    n_workers: 4
    
  resource_limits:
    memory_per_worker: "4GB"
    cpu_per_worker: 2
```

## Configuration Validation

```python
def validate_config(config: dict) -> None:
    """Validate configuration parameters"""
    
    # Validate data configuration
    assert 0 < config['lmlr']['correlation_threshold'] <= 1.0
    assert config['lmlr']['vif_threshold'] > 1.0
    assert config['lmlr']['max_iterations'] > 0
    
    # Validate deep learning configuration
    assert config['deep_learning']['epochs'] > 0
    assert config['deep_learning']['batch_size'] > 0
    assert 0 < config['deep_learning']['validation_split'] < 1.0
    assert config['deep_learning']['look_back'] > 0
    assert config['deep_learning']['forecast_horizon'] > 0
    
    # Validate causality configuration
    assert 0 < config['gcausal']['significance_level'] < 1.0
    assert config['gcausal']['max_lag_order'] > 0
    
    print("✅ Configuration validation passed")

# Usage
validate_config(config)
```

## Configuration Templates

Generate configuration templates for common use cases:

```python
def generate_config_template(use_case: str) -> str:
    """Generate configuration template for specific use case"""
    
    templates = {
        "healthcare": "config_templates/healthcare_template.yaml",
        "finance": "config_templates/finance_template.yaml",
        "iot": "config_templates/iot_template.yaml",
        "general": "config_templates/general_template.yaml"
    }
    
    template_path = templates.get(use_case, templates["general"])
    
    with open(template_path, 'r') as file:
        return file.read()

# Generate template
template = generate_config_template("healthcare")
with open("my_config.yaml", 'w') as file:
    file.write(template)
```

## Runtime Configuration Updates

```python
def update_config_runtime(config: dict, updates: dict) -> dict:
    """Update configuration at runtime"""
    
    def deep_update(base_dict, update_dict):
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict:
                deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    deep_update(config, updates)
    return config

# Usage
config = load_config('config.yaml')
runtime_updates = {
    'deep_learning': {
        'epochs': 200,
        'batch_size': 64
    }
}
config = update_config_runtime(config, runtime_updates)
```

## Configuration Best Practices

1. **Version Control**: Keep configuration files in version control
2. **Environment Separation**: Use different configs for dev/test/prod
3. **Documentation**: Document all configuration options
4. **Validation**: Always validate configuration before execution
5. **Defaults**: Provide sensible defaults for all parameters
6. **Modularity**: Split large configs into smaller, focused files

## Next Steps

- [Quick Start Guide](quickstart.md) - Apply configurations in practice
- [User Guide](../user-guide/overview.md) - Understand parameter impacts
- [Examples](../examples/advanced.md) - See configuration examples in action