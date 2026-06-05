# Configuration Reference

All Predap experiments are configured through the `BaseTransformerConfig` dataclass in `src/config/base_transformer_config.py`. This page documents every available parameter.

---

## Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lookback` | `int` | `14` | Number of past days used as model input |
| `forecast` | `int` | `7` | Number of days ahead to predict |
| `code` | `str` | `"J00"` | Target diagnostic code column name |

---

!!! warning "Lookback and Forecast Length"
    Your dataset must contain enough timestamps to cover the sum of the lookback and forecast windows.
    The minimum required dataset length is defined by the formula: Total Timestamps $\ge$ Lookback + Forecast.

## Architecture Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `head_size` | `int` | `64` | Dimensionality of each attention head |
| `num_heads` | `int` | `8` | Number of parallel attention heads |
| `ff_dim` | `int` | `512` | Hidden dimension of feed-forward network |
| `num_transformer_blocks` | `int` | `4` | Number of stacked Transformer encoder blocks |
| `mlp_units` | `list[int]` | `[512, 256]` | Hidden units in the MLP classification head |
| `dropout` | `float` | `0.25` | Dropout rate applied throughout the model |
| `activation_function` | `str` | `"gelu"` | Activation function (`gelu`, `relu`, `swish`) |
| `positional_encoding` | `bool` | `True` | Whether to apply sinusoidal positional encoding |

---

## Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `learning_rate` | `float` | `1e-4` | Base learning rate |
| `lr_max_multiplier` | `float` | `100` | Peak LR = `learning_rate × lr_max_multiplier` |
| `lr_min_multiplier` | `float` | `10` | Floor LR = `learning_rate × lr_min_multiplier` |
| `lr_warmup_ratio` | `float` | `0.2` | Fraction of total steps used for linear warmup |
| `epochs` | `int` | `4` | Maximum training epochs |
| `batch_size` | `int` | `256` | Training batch size (or use dynamic via `compute_dynamic_batch_size()`) |
| `early_stop_patience` | `int` | `50` | Epochs to wait before early stopping |
| `evaluate_model` | `bool` | `True` | Whether to run evaluation after training |

---

## Data Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_path` | `str` | — | Path to the `.parquet` or `.csv` dataset |
| `diagnostic_covariates_path` | `str` | — | Path prefix to `BEST_features_NOSMOOTH_` Excel files |
| `cutoff_date` | `str` | `"2008-01-01"` | Start date for data filtering |
| `final_cutoff_date` | `str` | `"2025-09-30"` | End date for data filtering |
| `default_split_ratio` | `float` | `0.8` | Train/test split ratio |
| `covid_token` | `bool` | `True` | Add binary COVID period indicator feature |
| `eliminate_covid_data` | `bool` | `False` | Remove COVID period rows from training |

---

## Seasonal Covariates

Default categorical variables used in Phase 3 (configurable via `DEFAULT_SEASONAL_CATEGORICAL_VARS`):

| Variable | Type |
|----------|------|
| `Day_of_Week` | Cyclical sin/cos |
| `Month` | Cyclical sin/cos |
| `Season` | Cyclical encoding |
| `Holiday` | Binary (Catalan calendar) |
| `School_Vacation` | Binary |
| `Is_Weekend` | Binary |

---

## Hyperparameter Search Lists

For grid search experiments, the config provides pre-defined lists:

| List | Default Values |
|------|---------------|
| `CODES_LIST` | All available diagnostic codes |
| `LOOKBACK_LIST` | `[7, 14, 60]` |
| `FORECAST_LIST` | `[7, 14, 30]` |
| `HEAD_SIZE_LIST` | `[32, 64]` |
| `NUM_HEADS_LIST` | `[4, 8, 16]` |
| `FF_DIM_LIST` | `[256, 512]` |
| `ACTIVATIONS_LIST` | `["gelu", "relu"]` |

---

## Hydra Configuration

Hydra YAML configs live in `conf/` and override `BaseTransformerConfig` values:

```yaml
# conf/grid_search_V1.yaml
model:
  target_code: "J00"
  lookback: 14
  forecast: 7
  head_size: 32
  num_heads: 8
  ff_dim: 512
  mlp_units: [512, 256]
  activation: "gelu"
  covid_token: true
  dropout: 0.5
  learning_rate: 1e-5

training:
  cutoff_date: "2008-01-01"
  positional_encoding: true

mlflow:
  tracking_uri: "file:./mlruns"
  experiment_name: "HYDRA_GRID_SEARCH"
```

Override on the command line:

```bash
python main_experiments_hydra.py --multirun \
    model.target_code=J00,B34 \
    model.lookback=7,14,60
```

---

## Model Name Generation

`BaseTransformerConfig` provides methods to generate structured model filenames:

```python
config = BaseTransformerConfig(code="J00", forecast=7, ff_dim=512, lookback=14, learning_rate=1e-5)

config.get_model_name()
# → "J00_base_transformer_7fh_512ff_14lb_1e-05lr.keras"

config.get_diagnostic_residual_model_name()
# → "J00_DIAGNOSTIC_RESIDUALS_LEARNING_7fh_14lb.keras"

config.get_seasonal_residual_model_name()
# → "J00_SEASONAL_RESIDUALS_LEARNING_7fh_14lb.keras"
```
