# Training Your First Model

This tutorial walks through training a complete three-phase Predap model for the diagnostic code `J00` (Acute nasopharyngitis) with a 7-day forecast horizon.

---

## Prerequisites

- Predap installed and configured (see [Installation](../getting-started/installation.md))
- A `.parquet` dataset with timestamped diagnostic visit counts
- Diagnostic covariate Excel files from the LMLR + GCausal pipeline

---

## Step 1: Import and Configure

```python
import mlflow
from datetime import datetime
from src.utils import experiments_utils

from src.main_train_univ_transformer_class import (
    TransformerUnivConfig,
    UnivariateTransformerPipeline,
)
from src.main_train_diagnostic_residual_transformer_class import (
    DiagnosticResidualTransformerConfig,
    DiagnosticResidualTransformerPipeline,
)
from src.main_train_seasonal_residual_transformer_class import (
    SeasonalResidualTransformerConfig,
    SeasonalResidualTransformerPipeline,
)
from src.univariate_transformer.utils_univ_transformer import load_mlflow_model_history

# Initialize MLflow tracking
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("my_first_experiment")
```

---

## Step 2: Define Experiment Parameters

```python
# Experiment parameters
CODE = "J00"
LOOKBACK = 14        # 14 days of history as input
FORECAST = 7         # Predict 7 days ahead
DATA_PATH = "data/full_CAT1.parquet"
CUTOFF_DATE = "2008-01-01"

# Model architecture
HEAD_SIZE = 32
NUM_HEADS = 8
FF_DIM = 512
MLP_UNITS = [512, 256]
ACTIVATION = "gelu"
LEARNING_RATE = 1e-5
COVID_TOKEN = True

# Compute optimal batch size based on sequence length
BATCH_SIZE = experiments_utils.compute_dynamic_batch_size(LOOKBACK, FORECAST)
print(f"Using batch size: {BATCH_SIZE}")
```

---

## Step 3: Phase 1 — Univariate Transformer

Train the baseline forecasting model:

```python
# Create configuration
univ_config = TransformerUnivConfig(
    lookback=LOOKBACK,
    forecast=FORECAST,
    code=CODE,
    activation_function=ACTIVATION,
    covid_token=COVID_TOKEN,
    cutoff_date=CUTOFF_DATE,
    head_size=HEAD_SIZE,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
    mlp_units=MLP_UNITS,
    evaluate_model=True,
    positional_encoding=False,
    data_path=DATA_PATH,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
)

# Run the pipeline
univ_pipeline = UnivariateTransformerPipeline(univ_config)
univ_outputs = univ_pipeline.run_complete_pipeline()

print(f"Phase 1 Results:")
print(f"  Model: {univ_outputs.model_name}")
print(f"  MSE:  {univ_outputs.mse:.4f}")
print(f"  MAE:  {univ_outputs.mae:.4f}")
print(f"  RMSE: {univ_outputs.rmse:.4f}")
print(f"  WAPE: {univ_outputs.wape:.4f}")
```

!!! tip "What happens internally"
    The pipeline: loads data → creates temporal features → builds a Transformer encoder with RevIN → trains with cosine LR schedule → evaluates on the test set → saves the `.keras` model and training history.

---

## Step 4: Phase 2 — Diagnostic Residual Correction

Refine the baseline by modeling residuals with diagnostic covariates:

```python
diag_config = DiagnosticResidualTransformerConfig(
    lookback=LOOKBACK,
    forecast=FORECAST,
    code=CODE,
    activation_function=ACTIVATION,
    covid_token=COVID_TOKEN,
    cutoff_date=CUTOFF_DATE,
    predictions_train_corrected=None,    # First residual phase
    predictions_test_corrected=None,
    head_size=HEAD_SIZE,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
    mlp_units=MLP_UNITS,
    evaluate_model=True,
    positional_encoding=False,
    data_path=DATA_PATH,
    learning_rate=1e-4,    # Higher LR for residual learning
    batch_size=BATCH_SIZE,
)

diag_pipeline = DiagnosticResidualTransformerPipeline(diag_config)
diag_outputs = diag_pipeline.run_complete_pipeline()

print(f"\nPhase 2 Results (after diagnostic correction):")
print(f"  MSE:  {diag_outputs.corrected_diagnostics_mse:.4f}")
print(f"  MAE:  {diag_outputs.corrected_diagnostics_mae:.4f}")
print(f"  RMSE: {diag_outputs.corrected_diagnostics_rmse:.4f}")
print(f"  WAPE: {diag_outputs.corrected_diagnostics_wape:.4f}")
```

---

## Step 5: Phase 3 — Seasonal Residual Correction

Apply the final correction using calendar/seasonal covariates:

```python
seasonal_config = SeasonalResidualTransformerConfig(
    lookback=LOOKBACK,
    forecast=FORECAST,
    code=CODE,
    activation_function=ACTIVATION,
    covid_token=COVID_TOKEN,
    cutoff_date=CUTOFF_DATE,
    predictions_train_corrected=diag_outputs.predictions_train_corrected,
    predictions_test_corrected=diag_outputs.predictions_test_corrected,
    batch_size=BATCH_SIZE,
)

seasonal_pipeline = SeasonalResidualTransformerPipeline(seasonal_config)
seasonal_outputs = seasonal_pipeline.run_complete_pipeline()

print(f"\nPhase 3 Results (final forecast):")
print(f"  MSE:  {seasonal_outputs.corrected_diagnostics_mse:.4f}")
print(f"  MAE:  {seasonal_outputs.corrected_diagnostics_mae:.4f}")
print(f"  RMSE: {seasonal_outputs.corrected_diagnostics_rmse:.4f}")
print(f"  WAPE: {seasonal_outputs.corrected_diagnostics_wape:.4f}")
```

---

## Step 6: Compare Results Across Phases

```python
print("\n" + "=" * 50)
print("RESULTS COMPARISON")
print("=" * 50)
print(f"{'Phase':<25} {'MSE':>10} {'MAE':>10} {'RMSE':>10}")
print("-" * 55)
print(f"{'1. Univariate':<25} {univ_outputs.mse:>10.4f} {univ_outputs.mae:>10.4f} {univ_outputs.rmse:>10.4f}")
print(f"{'2. + Diagnostic':<25} {diag_outputs.corrected_diagnostics_mse:>10.4f} {diag_outputs.corrected_diagnostics_mae:>10.4f} {diag_outputs.corrected_diagnostics_rmse:>10.4f}")
print(f"{'3. + Seasonal (Final)':<25} {seasonal_outputs.corrected_diagnostics_mse:>10.4f} {seasonal_outputs.corrected_diagnostics_mae:>10.4f} {seasonal_outputs.corrected_diagnostics_rmse:>10.4f}")
```

---

## Step 7: View in MLflow

```bash
mlflow ui --backend-store-uri file:./mlruns --port 5000
```

Navigate to [http://localhost:5000](http://localhost:5000) to see:

- All logged parameters (architecture, training, data configs)
- Per-phase metrics and timing
- Saved model artifacts (`.keras` files)
- Training history plots (loss curves, etc.)

---

## Generated Outputs

After training, you will find:

| Output | Location | Description |
|--------|----------|-------------|
| Model weights | `models/{code}_cat_model/` | `.keras` files per phase |
| Training plots | `plots/` | Loss curves, prediction plots |
| MLflow artifacts | `mlruns/` | Full experiment tracking data |
| Training history | `models/` | `.pkl` files with epoch-by-epoch metrics |
