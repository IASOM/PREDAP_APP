# Grid Search & Hyperparameter Tuning

Predap provides two approaches for systematic hyperparameter search: a Python-based grid search script and a Hydra-based configuration-driven runner.

---

## Approach 1: Python Grid Search Script

The `main_grid_search_hyperparameters.py` script iterates over all combinations of codes, lookback windows, and forecast horizons. Each combination runs the full three-phase pipeline with MLflow tracking.

### Configuration

Edit the script directly to define search spaces:

```python
from src.config.base_transformer_config import BaseTransformerConfig

default_config = BaseTransformerConfig()

# Search dimensions
CODES_LIST = ["demanda__TOTAL", "demanda__SERVEI_CODI__URG", "B34", "J00", "I10", "M54"]
LOOKBACK_LIST = default_config.LOOKBACK_LIST   # e.g., [7, 14, 60]
FORECAST_LIST = default_config.FORECAST_LIST   # e.g., [7, 14, 30]

# Fixed architecture (vary one thing at a time)
head_size = 32
num_heads = 8
ff_dim = 512
mlp_units = [512, 256]
activation = "gelu"
learning_rate = 1e-5
covid_token = True
```

### Running

```bash
python main_grid_search_hyperparameters.py
```

### Output

Results are tracked in:

- **MLflow** — Every run is logged with full params, metrics, and model artifacts
- **`best_hyperparameters_results/`** — Best configuration per diagnostic code

---

## Approach 2: Hydra Configuration Sweeps

The `main_experiments_hydra.py` script uses [Hydra](https://hydra.cc/) for declarative configuration and multi-run sweeps.

### Configuration Files

All Hydra configs are located in `conf/`:

| File | Purpose |
|------|---------|
| `config.yaml` | Single-run default configuration |
| `grid_search.yaml` | Multi-code architecture sweep |
| `grid_search_V1.yaml` | Experiment-pair based sweep with joblib parallelism |
| `grid_search_percentiles.yaml` | Percentile-based diagnostic code sweep |

### Experiment Pairs

`grid_search_V1.yaml` defines pre-configured `(lookback, forecast)` pairs:

```yaml
experiment_pairs:
  3_3:     { lb: 3,   fc: 3 }
  7_7:     { lb: 7,   fc: 7 }
  14_14:   { lb: 14,  fc: 14 }
  60_30:   { lb: 60,  fc: 30 }
  60_60:   { lb: 60,  fc: 60 }
  182_182: { lb: 182, fc: 182 }
  365_182: { lb: 182, fc: 365 }
```

### Running a Single Experiment

```bash
python main_experiments_hydra.py \
    model.target_code=J00 \
    experiment_setup=14_14
```

### Running a Multi-Run Sweep

```bash
python main_experiments_hydra.py --multirun \
    model.target_code=J00,B34,I10,M54 \
    experiment_setup=7_7,14_14,60_30
```

This produces $4 \times 3 = 12$ runs in total.

### Parallel Execution with Joblib

`grid_search_V1.yaml` is preconfigured with the `joblib` launcher:

```yaml
hydra:
  launcher:
    n_jobs: 1        # Set > 1 for parallel runs (requires sufficient GPU memory)
    backend: loky
  sweep:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

!!! warning "GPU Memory"
    When using `n_jobs > 1`, ensure each parallel run has enough GPU memory. Use `compute_dynamic_batch_size()` and consider reducing `batch_size` depending on the input data size(forecast and lookback).

### Overriding Any Parameter

```bash
# Override architecture
python main_experiments_hydra.py --multirun \
    model.head_size=32,64 \
    model.num_heads=4,8 \
    model.ff_dim=256,512

# Override training
python main_experiments_hydra.py \
    model.learning_rate=1e-4 \
    model.dropout=0.3 \
    training.cutoff_date=2010-01-01
```

### Output Structure

Hydra saves outputs in timestamped directories:

```
outputs/
└── 2026-03-03/
    └── 14-30-00/
        ├── .hydra/
        │   ├── config.yaml       # Resolved config for this run
        │   ├── hydra.yaml        # Hydra meta-config
        │   └── overrides.yaml    # CLI overrides used
        └── main_experiments_hydra.log
```

---

## Comparing Results

### MLflow Dashboard

Launch the MLflow UI to compare runs side-by-side:

```bash
mlflow ui --backend-store-uri file:./mlruns --port 5000
```

Key comparison metrics:

| Metric | Description |
|--------|-------------|
| `eval/residual_seasonal_model_mse` | Final MSE (after all 3 phases) |
| `eval/residual_seasonal_model_wape` | Final WAPE |
| `total_training_duration_minutes` | Total compute time |

### Dynamic Batch Size

Predap automatically computes an optimal batch size based on `lookback` and `forecast`:

```python
from src.utils.experiments_utils import compute_dynamic_batch_size

batch_size = compute_dynamic_batch_size(lookback=60, forecast=30)
# Returns 32–2048 based on GPU availability and sequence length
```

This prevents out-of-memory errors for long sequences while maximizing throughput for short ones. However the values of this function can be changed based on your machine GPU and RAM memory limitations. 
