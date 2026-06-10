# PREDAP Data Paths Configuration

## Overview

All data paths in the PREDAP pipeline have been corrected and unified to use real data from the AQUAS_DATA_RETRIEVAL module. This document describes the data flow across all pipeline stages.

---

## Data Architecture

```
PREDAP_APP/
├── AQUAS_DATA_RETRIEVAL/              ← DATA SOURCE
│   └── AQUAS_DATA_RETRIEVAL-main/
│       └── data/
│           ├── input/                 ← Raw data from AQUAS API
│           ├── output/                ← Intermediate processed data
│           └── finals/                ← Final prepared datasets
│               └── demand_diagnosis_joined.parquet  ← MAIN INPUT FILE
│
├── TRANSFORMERS_PREDAP/               ← PROCESSING PIPELINE
│   ├── data/
│   │   └── best_features/             ← Feature selection files (XLSX)
│   ├── quantized_models/              ← OUTPUT: Quantized model weights
│   ├── production_predictions/        ← OUTPUT: Predictions & metrics
│   ├── plots/plots_residual_transformers/ ← OUTPUT: Visualization plots
│   └── mlruns/                        ← MLflow experiment tracking
│
└── Inference/                         ← STANDALONE INFERENCE BUNDLE
    ├── config/
    ├── production/
    └── utils/
```

---

## Main Data Paths

### 1. Input Data (Training & Inference)

**Primary Data Source:**
```
AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/data/finals/demand_diagnosis_joined.parquet
```

**Used by:**
- Training phase (univariate, diagnostic, seasonal models)
- Quantization phase (model loading & evaluation)
- Inference/Reconstruction phase
- All model evaluation and metrics computation

**Format:** Apache Parquet (efficient columnar format)
**Contents:** Time series demand data with diagnostic features and seasonal patterns
**Time Range:** 2008-01-01 to 2025-12-31

**Configuration references:**
```yaml
# conf/config.yaml
data:
  data_path: "AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/data/finals/demand_diagnosis_joined.parquet"

# conf/config_production.yaml
data:
  data_path: 'AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/data/finals/demand_diagnosis_joined.parquet'

# src/config/base_transformer_config.py
data_path: str = 'AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/data/finals/demand_diagnosis_joined.parquet'
```

---

### 2. Feature Selection / Diagnostic Covariates

**Path Pattern:**
```
data/best_features/BEST_features_NOSMOOTH_<CODE>.xlsx
```

**Example Files:**
- `data/best_features/BEST_features_NOSMOOTH_DEMANDA_TOTAL.xlsx`
- `data/best_features/BEST_features_NOSMOOTH_DEMANDA_SERVEI_CODI_INF.xlsx`
- `data/best_features/BEST_features_NOSMOOTH_DEMANDA_SERVEI_CODI_INFP.xlsx`

**Used by:**
- Training phase (diagnostic model feature selection)
- Quantization phase (evaluation with selected features)
- Inference/Reconstruction phase

**Format:** Excel spreadsheet with columns:
- `LAG`: Forecast horizon (7, 14, 30, 60, 182, 365)
- `predictors`: Comma-separated list of selected features

**Configuration references:**
```python
# src/config/base_transformer_config.py
diagnostic_covariates_path: str = 'data/best_features/BEST_features_NOSMOOTH_'

# Inference/config/base_transformer_config.py
diagnostic_covariates_path: str = "../../data/best_features/BEST_features_NOSMOOTH_"

# predap_cli.py (can be overridden)
--diagnostic-covariates-prefix  # CLI argument
```

---

### 3. Model Artifacts & Quantized Weights

**Quantized Models Output Path:**
```
quantized_models/<CODE>/<MODEL_TYPE>/<CODE>_<MODEL_TYPE>_<FORECAST>fh_<LOOKBACK>lb_f16_weights.h5
```

**Example Paths:**
```
quantized_models/DEMANDA_TOTAL/univariate_model/DEMANDA_TOTAL_univariate_model_7fh_7lb_f16_weights.h5
quantized_models/DEMANDA_TOTAL/residual_diagnostics_model/DEMANDA_TOTAL_residual_diagnostics_model_7fh_7lb_f16_weights.h5
quantized_models/DEMANDA_TOTAL/residual_seasonal_model/DEMANDA_TOTAL_residual_seasonal_model_7fh_7lb_f16_weights.h5
```

**Model Types:**
- `univariate_model`: Base univariate transformer (main forecast)
- `residual_diagnostics_model`: Residual model for diagnostic components
- `residual_seasonal_model`: Residual model for seasonal patterns

**MLflow Artifact Repository:**
```
mlruns/
```
Stores full model artifacts during training, which are loaded and quantized during the quantization phase.

**Configuration references:**
```python
# production/model_quantization_pipeline.py
save_weights_path = f"quantized_models/{code}/{model_name}/..."

# production/retrieve_and_reconstruct_data_pipeline.py
model_folder = "quantized_models"

# Inference/production/retrieve_and_reconstruct_data_pipeline.py (requires --model-folder argument)
```

---

### 4. Production Outputs

**Predictions Output:**
```
production_predictions/final_output_predictions.parquet
production_predictions/final_output_predictions/  # Directory with per-timestamp files
```

**Metrics Output:**
```
production_predictions/production_evaluation_metrics.parquet
```

**Used by:**
- Quantization phase (saves evaluation metrics)
- Inference/Reconstruction phase (saves predictions)
- Model comparison and validation

**Configuration references:**
```python
# src/config/base_transformer_config.py
production_predictions_dir: str = 'production_predictions/final_output_predictions'
production_predictions_file: str = 'production_predictions/final_output_predictions.parquet'
production_metrics_file: str = 'production_predictions/production_evaluation_metrics.parquet'

# Inference/config/base_transformer_config.py
production_predictions_dir: str = "../../production_predictions/final_output_predictions"
production_predictions_file: str = "../../production_predictions/final_output_predictions.parquet"
production_metrics_file: str = "../../production_predictions/production_evaluation_metrics.parquet"
```

---

### 5. Visualization Outputs

**Plot Output Directory:**
```
plots/plots_residual_transformers/
```

**Example Files:**
```
plots/plots_residual_transformers/predictions_with_waves_Model.png
plots/plots_residual_transformers/stepwise_errors_Model.png
plots/plots_residual_transformers/mse_over_time_Model.png
plots/plots_residual_transformers/mae_over_time_Model.png
plots/plots_residual_transformers/rmse_over_time_Model.png
plots/plots_residual_transformers/wape_over_time_Model.png
```

**Used by:**
- Training phase (training progress visualization)
- Quantization evaluation (performance comparison plots)

**Configuration references:**
```python
# src/utils/evaluation_plot_utils.py
PLOTS_DIR = Path("plots/plots_residual_transformers")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
```

---

## Pipeline Data Flow

### Phase 1: Training (3 Stages)

```
INPUT: demand_diagnosis_joined.parquet
       + BEST_features_NOSMOOTH_*.xlsx
              ↓
    ┌─────────────────────┐
    │ Stage 1: Univariate │
    │    Transformer      │
    └─────────────────────┘
              ↓
    ┌─────────────────────┐
    │ Stage 2: Diagnostic │
    │    Residuals        │
    └─────────────────────┘
              ↓
    ┌─────────────────────┐
    │ Stage 3: Seasonal   │
    │    Residuals        │
    └─────────────────────┘
              ↓
OUTPUT: models/ (MLflow artifacts)
        + training metrics
```

**Command:**
```bash
python predap_cli.py train \
  --codes TOTAL INFP MF \
  --lookbacks 7 14 60 \
  --forecasts 7 14 30 \
  --data-path "AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/data/finals/demand_diagnosis_joined.parquet"
```

**Data Path Used:**
- Input: `demand_diagnosis_joined.parquet`
- Diagnostic features: `BEST_features_NOSMOOTH_*.xlsx`
- Output: MLflow `mlruns/` directory

---

### Phase 2: Quantization

```
INPUT: demand_diagnosis_joined.parquet
       + trained models from mlruns/
       + BEST_features_NOSMOOTH_*.xlsx
              ↓
    ┌─────────────────────────────┐
    │ Load Models from MLflow     │
    │ (code canonicalization)     │
    └─────────────────────────────┘
              ↓
    ┌─────────────────────────────┐
    │ Quantize Weights to float16 │
    └─────────────────────────────┘
              ↓
    ┌─────────────────────────────┐
    │ Evaluate Quantization Impact│
    └─────────────────────────────┘
              ↓
OUTPUT: quantized_models/<CODE>/<TYPE>/*.h5
        + production_predictions/evaluation_metrics.parquet
        + plots/plots_residual_transformers/*.png
```

**Command:**
```bash
python predap_cli.py quantize \
  --codes TOTAL \
  --lookbacks 7 \
  --forecasts 7 \
  --experiments "1.0_Production_TRANSFORMER_TRANSFORMERS_PREDAP_HYDRA_GRID_SEARCH_20260514" \
  --evaluate
```

**Data Paths Used:**
- Input: `demand_diagnosis_joined.parquet`
- Model input: MLflow artifacts
- Output: `quantized_models/` directory

---

### Phase 3: Reconstruction & Inference

```
INPUT: demand_diagnosis_joined.parquet
       + quantized_models/<CODE>/**/*.h5
       + BEST_features_NOSMOOTH_*.xlsx
              ↓
    ┌─────────────────────────────────┐
    │ Load Quantized Models           │
    │ (from quantized_models/)        │
    └─────────────────────────────────┘
              ↓
    ┌─────────────────────────────────┐
    │ Prepare Test/Inference Data     │
    │ (from parquet + features)       │
    └─────────────────────────────────┘
              ↓
    ┌─────────────────────────────────┐
    │ Run Inference                   │
    │ (3-stage cascade)               │
    └─────────────────────────────────┘
              ↓
OUTPUT: production_predictions/final_output_predictions.parquet
        + production_predictions/production_evaluation_metrics.parquet
```

**Command (from main pipeline):**
```bash
python production/retrieve_and_reconstruct_data_pipeline.py
```

**Command (inference bundle - standalone):**
```bash
python Inference/production/retrieve_and_reconstruct_data_pipeline.py \
  --input-directory "AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/data/finals/demand_diagnosis_joined.parquet" \
  --old-input-directory "AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/data/finals/demand_diagnosis_joined.parquet" \
  --model-folder "quantized_models" \
  --output-path "production_predictions/final_output_predictions" \
  --metrics-df-path "production_predictions/production_evaluation_metrics.parquet" \
  --diagnostic-covariates-path "data/best_features/BEST_features_NOSMOOTH_"
```

**Data Paths Used:**
- Input: `demand_diagnosis_joined.parquet`
- Models: `quantized_models/`
- Features: `BEST_features_NOSMOOTH_*.xlsx`
- Output: `production_predictions/`

---

## Code Name Canonicalization

The system automatically converts user input code names to canonical form:

| User Input | Canonical Form | Description |
|------------|-----------------|-------------|
| `TOTAL` | `DEMANDA_TOTAL` | Total demand |
| `INF` | `DEMANDA_SERVEI_CODI_INF` | Primary care |
| `INFP` | `DEMANDA_SERVEI_CODI_INFP` | Primary + pediatric |
| `MF` | `DEMANDA_SERVEI_CODI_MEDFAM` | Family medicine |

**Applied in:**
1. CLI argument parsing (`predap_cli.py` → `_canonical_code_name()`)
2. MLflow run search (`model_quantization_pipeline.py` → `_canonicalize_code()`)
3. Data column lookup (`predap_cli.py` → `_filter_codes_in_dataset()`)

---

## Temporal Parameters

### Date Ranges

```python
# Training cutoff (historical data start)
cutoff_date: "2008-01-01"

# Maximum date (data period end)
max_date: "2025-12-31"

# COVID-19 periods (optional elimination)
covid_dates: [
    ("2020-03-01", "2020-06-30"),
    ("2020-10-01", "2020-12-31"),
    ("2021-01-01", "2021-03-31"),
    ("2021-04-01", "2021-06-30"),
]
```

### Temporal Windows

```python
# Lookback/Forecast pairs used in experiments
experiment_pairs: {
    "3_3":   { lb: 3,   fc: 3 },
    "7_7":   { lb: 7,   fc: 7 },
    "14_14": { lb: 14,  fc: 14 },
    "60_30": { lb: 60,  fc: 30 },
    "60_60": { lb: 60,  fc: 60 },
    "182_182": { lb: 182, fc: 182 },
    "365_182": { lb: 182, fc: 365 }
}
```

---

## Relative vs. Absolute Paths

### Main Pipeline (TRANSFORMERS_PREDAP root)

All paths are relative to `TRANSFORMERS_PREDAP/`:

```python
# ✅ CORRECT
'AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/data/finals/demand_diagnosis_joined.parquet'
'data/best_features/BEST_features_NOSMOOTH_'
'quantized_models/'
'production_predictions/'
'plots/plots_residual_transformers/'
```

### Inference Bundle (Inference subdirectory)

All paths are relative to `TRANSFORMERS_PREDAP/Inference/`, using `../../` to go up to root:

```python
# ✅ CORRECT (from Inference/)
'../../AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/data/finals/demand_diagnosis_joined.parquet'
'../../data/best_features/BEST_features_NOSMOOTH_'
'../../quantized_models/'
'../../production_predictions/'
```

---

## Environment Variables & Configuration

### Config Files

| File | Purpose | Data Paths |
|------|---------|-----------|
| `conf/config.yaml` | Default Hydra configuration | Input data path |
| `conf/config_production.yaml` | Production grid search config | Input data path, codes path |
| `src/config/base_transformer_config.py` | Main config class | All data paths (training/inference) |
| `Inference/config/base_transformer_config.py` | Inference-only config | All data paths (inference bundle) |

### MLflow Tracking

**Tracking URI:**
```yaml
# conf/config_production.yaml
mlflow:
  tracking_uri: "file:./mlruns"
  experiment_name: "1.0_Production_TRANSFORMER_TRANSFORMERS_PREDAP_HYDRA_GRID_SEARCH"
```

**Run Name Format:**
```
1.0_Production_TRANSFORMER_<CODE>_lb<LOOKBACK>_fh<FORECAST>_<TIMESTAMP>
```

**Example:**
```
1.0_Production_TRANSFORMER_DEMANDA_TOTAL_lb7_fh7_120530
```

---

## Troubleshooting

### Issue: "No such file or directory"

**Check:**
1. Are you running from the correct directory? (TRANSFORMERS_PREDAP root)
2. Does `AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/data/finals/demand_diagnosis_joined.parquet` exist?
3. For feature files, do `data/best_features/BEST_features_NOSMOOTH_<CODE>.xlsx` files exist?

### Issue: "No run found starting with..."

**Check:**
1. Has training been completed for the specified code, lookback, forecast?
2. Is the code canonicalized correctly? (TOTAL → DEMANDA_TOTAL)
3. Are you using the correct experiment name?
4. Check MLflow: `mlruns/<EXPERIMENT_ID>/`

### Issue: "Plot directory missing"

**Check:**
1. Directory `plots/plots_residual_transformers/` is auto-created
2. If error persists, ensure write permissions to project root
3. Check file paths are relative to TRANSFORMERS_PREDAP root

### Issue: Missing features in diagnosis

**Check:**
1. Does `data/best_features/BEST_features_NOSMOOTH_<CODE>.xlsx` exist?
2. File must contain columns: `LAG`, `predictors`
3. LAG value must match forecast horizon (7, 14, 30, 60, 182, 365)

---

## Summary of Changes

### Files Modified

1. **conf/config.yaml**
   - ✅ Updated data_path to point to AQUAS output

2. **conf/config_production.yaml**
   - ✅ Updated data_path to AQUAS output
   - ✅ Updated codes_path to AQUAS location

3. **src/config/base_transformer_config.py**
   - ✅ Fixed all data paths to use AQUAS output
   - ✅ Added production output paths
   - ✅ Removed duplicate entries

4. **production/model_quantization_pipeline.py**
   - ✅ Fixed quantized_models output path
   - ✅ Added code canonicalization
   - ✅ Added MLflow search diagnostics

5. **production/retrieve_and_reconstruct_data_pipeline.py**
   - ✅ Updated all hardcoded paths to AQUAS output
   - ✅ Fixed output directory paths

6. **Inference/config/base_transformer_config.py**
   - ✅ Updated all data paths with correct relative paths

---

## Next Steps

1. **Generate AQUAS Data:**
   ```bash
   cd AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main
   python run_pipeline.py  # Or follow AQUAS_DATA_RETRIEVAL README
   ```

2. **Train Models:**
   ```bash
   cd TRANSFORMERS_PREDAP
   python predap_cli.py train --codes TOTAL --lookbacks 7 --forecasts 7
   ```

3. **Quantize Models:**
   ```bash
   python predap_cli.py quantize --codes TOTAL --lookbacks 7 --forecasts 7
   ```

4. **Run Inference:**
   ```bash
   python production/retrieve_and_reconstruct_data_pipeline.py
   ```

---

**Last Updated:** 2026-06-10
**Status:** ✅ All data paths corrected and unified
