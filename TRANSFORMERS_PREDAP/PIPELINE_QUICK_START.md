# PREDAP Complete Pipeline Quick Start Guide

## Overview

This guide walks you through running the complete PREDAP forecasting pipeline with corrected data paths:

1. **Training Phase** (3 stages): Univariate → Diagnostic → Seasonal models
2. **Quantization Phase**: Load and quantize trained models
3. **Reconstruction & Inference**: Generate predictions on test data

---

## Prerequisites

### 1. Data Generation (AQUAS)

Before running the pipeline, generate data from AQUAS:

```bash
cd AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main

# Follow the README or run the pipeline
python run_pipeline.py

# This should create:
# data/finals/demand_diagnosis_joined.parquet
# (This is the main input file for all pipeline stages)
```

### 2. Feature Files

Ensure feature selection XLSX files exist:

```bash
ls data/best_features/BEST_features_NOSMOOTH_*.xlsx

# Example files needed:
# data/best_features/BEST_features_NOSMOOTH_DEMANDA_TOTAL.xlsx
# data/best_features/BEST_features_NOSMOOTH_DEMANDA_SERVEI_CODI_INF.xlsx
# etc.
```

### 3. Verify Paths

Run the verification script:

```bash
cd TRANSFORMERS_PREDAP
python verify_data_paths.py
```

Expected output:
```
✓ ALL REQUIRED PATHS VERIFIED SUCCESSFULLY
```

---

## Pipeline Execution

### Option A: Complete End-to-End Pipeline

Run all stages automatically:

```bash
cd TRANSFORMERS_PREDAP

# 1. TRAINING (3 phases): ~2-4 hours depending on data size
python predap_cli.py train \
  --codes TOTAL \
  --lookbacks 7 \
  --forecasts 7 \
  --stage univariate

python predap_cli.py train \
  --codes TOTAL \
  --lookbacks 7 \
  --forecasts 7 \
  --stage diagnostic

python predap_cli.py train \
  --codes TOTAL \
  --lookbacks 7 \
  --forecasts 7 \
  --stage seasonal

# 2. QUANTIZATION: ~15-30 minutes
python predap_cli.py quantize \
  --codes TOTAL \
  --lookbacks 7 \
  --forecasts 7 \
  --experiments "1.0_Production_TRANSFORMER_TRANSFORMERS_PREDAP_HYDRA_GRID_SEARCH_20260514" \
  --evaluate

# 3. INFERENCE/RECONSTRUCTION: ~10-20 minutes
python production/retrieve_and_reconstruct_data_pipeline.py
```

### Option B: Individual Stages

#### Step 1: Train Univariate Model

```bash
python predap_cli.py train \
  --codes TOTAL INF INFP MF \
  --lookbacks 7 14 60 \
  --forecasts 7 14 30 \
  --stage univariate \
  --data-path "AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/data/finals/demand_diagnosis_joined.parquet" \
  --epochs 200 \
  --batch-size 256
```

**Output:**
- MLflow artifacts in `mlruns/`
- Training logs in `outputs/`

**Check MLflow:**
```bash
# View training progress
mlflow ui --backend-store-uri="file:./mlruns"
# Open http://localhost:5000
```

#### Step 2: Train Diagnostic Residual Model

```bash
python predap_cli.py train \
  --codes TOTAL INF INFP MF \
  --lookbacks 7 14 60 \
  --forecasts 7 14 30 \
  --stage diagnostic \
  --data-path "AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/data/finals/demand_diagnosis_joined.parquet"
```

**Output:**
- Diagnostic residual models in MLflow
- Evaluation plots in `plots/plots_residual_transformers/`

#### Step 3: Train Seasonal Residual Model

```bash
python predap_cli.py train \
  --codes TOTAL INF INFP MF \
  --lookbacks 7 14 60 \
  --forecasts 7 14 30 \
  --stage seasonal \
  --data-path "AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/data/finals/demand_diagnosis_joined.parquet"
```

**Output:**
- Seasonal residual models in MLflow
- Training complete!

---

## Phase 2: Quantization

After training completes, quantize the models:

```bash
# Get experiment ID from MLflow
export EXP_ID="1.0_Production_TRANSFORMER_TRANSFORMERS_PREDAP_HYDRA_GRID_SEARCH_20260514"

python predap_cli.py quantize \
  --codes TOTAL INF INFP MF \
  --lookbacks 7 14 60 \
  --forecasts 7 14 30 \
  --experiments "$EXP_ID" \
  --evaluate \
  --data-path "AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/data/finals/demand_diagnosis_joined.parquet"
```

**Output:**
- Quantized model weights: `quantized_models/<CODE>/<TYPE>/*_f16_weights.h5`
- Evaluation metrics: `production_predictions/production_evaluation_metrics.parquet`
- Comparison plots: `plots/plots_residual_transformers/`

**Verify:**
```bash
ls quantized_models/DEMANDA_TOTAL/*/
```

Expected output:
```
univariate_model/:
  DEMANDA_TOTAL_univariate_model_7fh_7lb_f16_weights.h5

residual_diagnostics_model/:
  DEMANDA_TOTAL_residual_diagnostics_model_7fh_7lb_f16_weights.h5

residual_seasonal_model/:
  DEMANDA_TOTAL_residual_seasonal_model_7fh_7lb_f16_weights.h5
```

---

## Phase 3: Reconstruction & Inference

Run predictions on test data using quantized models:

```bash
python production/retrieve_and_reconstruct_data_pipeline.py
```

**Output:**
- Predictions: `production_predictions/final_output_predictions.parquet`
- Metrics: `production_predictions/production_evaluation_metrics.parquet`
- Individual predictions: `production_predictions/final_output_predictions/`

**Verify:**
```bash
ls production_predictions/

# Expected files:
# - production_evaluation_metrics.parquet
# - final_output_predictions.parquet
# - final_output_predictions/ (directory with per-timestamp files)
```

---

## Standalone Inference (Inference Bundle)

If you have the `Inference/` standalone bundle:

```bash
cd Inference

python production/retrieve_and_reconstruct_data_pipeline.py \
  --input-directory "../../AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/data/finals/demand_diagnosis_joined.parquet" \
  --old-input-directory "../../AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/data/finals/demand_diagnosis_joined.parquet" \
  --model-folder "../../quantized_models" \
  --output-path "../../production_predictions/final_output_predictions" \
  --metrics-df-path "../../production_predictions/production_evaluation_metrics.parquet" \
  --diagnostic-covariates-path "../../data/best_features/BEST_features_NOSMOOTH_" \
  --lookback-list "7,14,60,60,182,182" \
  --forecast-list "7,14,30,60,182,365"
```

---

## Key Data Paths Reference

| Stage | Input | Output | Path |
|-------|-------|--------|------|
| **Training** | `demand_diagnosis_joined.parquet` | MLflow artifacts | `AQUAS_DATA_RETRIEVAL/.../data/finals/` → `mlruns/` |
| **Quantization** | MLflow artifacts + input data | Quantized weights | `mlruns/` → `quantized_models/` |
| **Inference** | Quantized weights + input data | Predictions | `quantized_models/` → `production_predictions/` |

---

## Data Path Configuration

All paths are configured in:

1. **conf/config.yaml** - Default settings
2. **conf/config_production.yaml** - Production settings
3. **src/config/base_transformer_config.py** - Python config class

### Main Input Data

```yaml
# conf/config.yaml and conf/config_production.yaml
data:
  data_path: "AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/data/finals/demand_diagnosis_joined.parquet"
```

### Diagnostic Features

```yaml
# Located in data/best_features/
# Pattern: BEST_features_NOSMOOTH_<CODE>.xlsx
# Example: BEST_features_NOSMOOTH_DEMANDA_TOTAL.xlsx
```

### Outputs

```
quantized_models/                          # Quantized model weights
production_predictions/                    # Predictions & metrics
plots/plots_residual_transformers/        # Training/evaluation plots
mlruns/                                   # MLflow experiments
```

---

## Troubleshooting

### "No such file or directory: demand_diagnosis_joined.parquet"

**Solution:**
1. Generate AQUAS data first
2. Verify file exists: `ls AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/data/finals/`
3. Run from TRANSFORMERS_PREDAP root directory

### "No run found starting with..."

**Solution:**
1. Ensure training was completed for the code/lookback/forecast
2. Check code canonicalization: `TOTAL` → `DEMANDA_TOTAL`
3. Verify experiment name matches: `mlflow ui` to find correct name

### "No such file or directory: BEST_features_NOSMOOTH_*.xlsx"

**Solution:**
1. Feature files must be in: `data/best_features/`
2. Format: `BEST_features_NOSMOOTH_<CODE>.xlsx`
3. Generate from grid search or create manually

### "Plots directory missing"

**Solution:**
Directory is auto-created. If error persists:
1. Check write permissions
2. Manually create: `mkdir -p plots/plots_residual_transformers`

---

## Monitoring & Diagnostics

### MLflow Dashboard

```bash
mlflow ui --backend-store-uri="file:./mlruns"
# Open http://localhost:5000
```

View:
- Training experiments and runs
- Model artifacts
- Metrics and parameters
- Training logs

### Verify Paths Script

```bash
python verify_data_paths.py
```

Checks:
- Input data exists
- Output directories exist or will be created
- Configuration files are valid
- Python imports work correctly

### View Quantized Models

```bash
# List quantized weights
find quantized_models/ -name "*_f16_weights.h5"

# Check file size (quantized models should be ~3-5MB each)
du -h quantized_models/
```

### Check Predictions Output

```bash
# View metrics
python -c "import pandas as pd; print(pd.read_parquet('production_predictions/production_evaluation_metrics.parquet'))"

# View sample predictions
python -c "import pandas as pd; print(pd.read_parquet('production_predictions/final_output_predictions.parquet').head())"
```

---

## Performance Tips

### Reduce Training Time

```bash
# Train on fewer codes
python predap_cli.py train --codes TOTAL --stage univariate

# Reduce epochs
python predap_cli.py train --codes TOTAL --epochs 50 --stage univariate

# Use smaller batch size for faster iterations
python predap_cli.py train --codes TOTAL --batch-size 64 --stage univariate
```

### Parallel Training

```bash
# Train multiple temporal windows in parallel (use nohup or tmux)
nohup python predap_cli.py train --codes TOTAL --lookbacks 7 --forecasts 7 --stage univariate &
nohup python predap_cli.py train --codes INF --lookbacks 14 --forecasts 14 --stage univariate &
```

### Monitor Resources

```bash
# Linux/Mac
watch -n 1 'nvidia-smi'  # GPU usage
top                       # CPU/Memory

# Windows PowerShell
Get-Process python | Format-Table ProcessName, CPU, Memory
```

---

## Complete Workflow Example

```bash
#!/bin/bash
# Full pipeline example

cd /path/to/TRANSFORMERS_PREDAP

# 1. Verify paths
echo "=== Verifying paths ==="
python verify_data_paths.py

# 2. Train models (all stages)
echo "=== Training univariate model ==="
python predap_cli.py train --codes TOTAL --lookbacks 7 --forecasts 7 --stage univariate

echo "=== Training diagnostic model ==="
python predap_cli.py train --codes TOTAL --lookbacks 7 --forecasts 7 --stage diagnostic

echo "=== Training seasonal model ==="
python predap_cli.py train --codes TOTAL --lookbacks 7 --forecasts 7 --stage seasonal

# 3. Quantize models
echo "=== Quantizing models ==="
python predap_cli.py quantize \
  --codes TOTAL \
  --lookbacks 7 \
  --forecasts 7 \
  --experiments "1.0_Production_TRANSFORMER_TRANSFORMERS_PREDAP_HYDRA_GRID_SEARCH_20260514" \
  --evaluate

# 4. Run inference
echo "=== Running inference ==="
python production/retrieve_and_reconstruct_data_pipeline.py

# 5. Verify outputs
echo "=== Pipeline complete! Checking outputs ==="
echo "Quantized models:"
ls -lh quantized_models/DEMANDA_TOTAL/*/
echo "Predictions:"
ls -lh production_predictions/*.parquet
echo "Done!"
```

---

## Next Steps

1. ✅ Generate AQUAS data
2. ✅ Run verification script
3. ✅ Train models (Phase 1)
4. ✅ Quantize models (Phase 2)
5. ✅ Run inference (Phase 3)
6. 📊 Analyze results
7. 🚀 Deploy to production

---

**Last Updated:** 2026-06-10
**Status:** ✅ All data paths verified and tested
