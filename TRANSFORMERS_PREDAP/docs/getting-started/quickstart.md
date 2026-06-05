# 5-Minute Quickstart

Get an end-to-end training run working in under 5 minutes.

---

## 1. Configure Your Experiment

All parameters are centralized in `BaseTransformerConfig`. You can instantiate directly:

```python
from src.config.base_transformer_config import BaseTransformerConfig

config = BaseTransformerConfig()
print(config.data_path)       # Path to your .parquet dataset
print(config.CODES_LIST)      # Available diagnostic codes
print(config.LOOKBACK_LIST)   # e.g., [7, 14, 60]
print(config.FORECAST_LIST)   # e.g., [7, 14, 30]
```

---

## 2. Run the Three-Phase Pipeline

The simplest way to run the full pipeline is via `main.py`:

```bash
python main.py
```

This iterates over every `(code, lookback, forecast)` combination and, for each, executes:

1. **Univariate Transformer** — baseline forecast
2. **Diagnostic Residual Transformer** — residual correction with diagnostic covariates
3. **Seasonal Residual Transformer** — further correction with calendar features

All runs are tracked in MLflow automatically.

---

## 3. Run a Single Experiment Programmatically

```python
from src.main_train_univ_transformer_class import (
    TransformerUnivConfig,
    UnivariateTransformerPipeline,
)

# Define configuration
config = TransformerUnivConfig(
    lookback=14,
    forecast=7,
    code="J00",
    activation_function="gelu",
    covid_token=True,
    cutoff_date="2008-01-01",
    head_size=32,
    num_heads=8,
    ff_dim=512,
    mlp_units=[512, 256],
    evaluate_model=True,
    positional_encoding=False,
    data_path="path/to/your/data.parquet",
    learning_rate=1e-5,
    batch_size=256,
)

# Run the complete pipeline
pipeline = UnivariateTransformerPipeline(config)
outputs = pipeline.run_complete_pipeline()

# Access results
print(f"Model: {outputs.model_name}")
print(f"MSE:   {outputs.mse:.4f}")
print(f"MAE:   {outputs.mae:.4f}")
print(f"RMSE:  {outputs.rmse:.4f}")
print(f"WAPE:  {outputs.wape:.4f}")
```

---

## 4. Use the Grid Search Runner

For hyperparameter sweeps, use the Hydra-based runner:

```bash
python main_experiments_hydra.py --multirun \
    model.target_code=J00,B34,I10 \
    experiment_setup=7_7,14_14,60_30
```

Or use the built-in grid search script:

```bash
python main_grid_search_hyperparameters.py
```

---

## 5. View Results

Launch MLflow to inspect tracked metrics, artifacts, and models:

```bash
mlflow ui --backend-store-uri file:./mlruns --port 5000
```

Key metrics logged per run:

| Metric | Description |
|--------|-------------|
| `eval/univ_transformer_mse` | Univariate model MSE |
| `eval/residual_diagnostics_model_rmse` | Diagnostic residual RMSE |
| `eval/residual_seasonal_model_wape` | Final seasonal WAPE |
| `total_training_duration_minutes` | End-to-end duration |

---

## 6. Start the Production API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

!!! tip "Next Steps"
    - Read the [Model Training Guide](../user-guide/model-training.md) for an in-depth walkthrough
    - Explore [Configuration Reference](../user-guide/configuration.md) for all available parameters
    - Check the [Tutorials](../tutorials/train-custom-model.md) for step-by-step examples
