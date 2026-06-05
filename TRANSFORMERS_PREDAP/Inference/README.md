# Inference

This folder is a self-contained inference bundle for `retrieve_and_reconstruct_data_pipeline`.

## Contents

- `production/` contains the runnable inference pipeline.
- `config/`, `utils/`, `univariate_transformer/`, and `residual_multivariate_transformers/` contain the code the pipeline imports.
- `requirements.txt` and `Dockerfile` are scoped to this folder only.

## Build

```bash
docker build -t predap-inference .
```

## Run

The pipeline expects mounted model weights and input data. Example:

```bash
docker run --rm \
  -v "$PWD/data:/app/data" \
  -v "$PWD/models:/app/models" \
  -v "$PWD/output:/app/output" \
  predap-inference \
  --input-directory /app/data/demand_diagnostics_joined.parquet \
  --old-input-directory /app/data/finals_combined.csv \
  --model-folder /app/models \
  --output-path /app/output/final_output_predictions \
  --metrics-df-path /app/output/production_evaluation_metrics.parquet
```

The default pipeline entrypoint is `production/retrieve_and_reconstruct_data_pipeline.py`.
