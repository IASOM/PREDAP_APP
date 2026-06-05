# Pydantic Models

This page documents the request validation models used by the FastAPI service.

Source file: `api/schemas/production_schemas.py`

## AddNewDataRequest

Used by `POST /production/add_new_data`.

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `new_data_path` | `str \| None` | No | `None` | Path to the source parquet dataset. Falls back to server config when omitted. |
| `cutoff_date` | `str \| None` | No | `None` | Lower date boundary for processing. Falls back to server config when omitted. |
| `max_date` | `str \| None` | No | `None` | Upper date boundary for processing. Falls back to server config when omitted. |
| `eliminate_covid_data` | `bool \| None` | No | `None` | If true, excludes COVID years from mean calculations. Falls back to server config when omitted. |
| `covid_token` | `bool \| None` | No | `None` | If true, keeps/creates the COVID indicator token. Falls back to server config when omitted. |
| `provided_data` | `list[float] \| None` | No | `None` | Explicit values for the new row. Length should match the expected feature count. |
| `save_path` | `str \| None` | No | `"../data/FINAL_DB1"` | Output folder for the updated dataset. |
| `delete_old` | `bool \| None` | No | `True` | If true, existing file with matching name is removed before save. |

Example payload:

```json
{
  "new_data_path": "../data/FINAL_DB/full_CAT1.parquet",
  "cutoff_date": "2008-01-01",
  "max_date": "2025-09-30",
  "eliminate_covid_data": true,
  "covid_token": true,
  "provided_data": [10.4, 11.2, 9.8],
  "save_path": "../data/FINAL_DB1",
  "delete_old": true
}
```

## ModelReconstructionRequest

Used by `POST /production/model_reconstruction_pipeline`.

### Required fields

| Field | Type | Description |
|---|---|---|
| `code` | `str` | Target diagnostic/service code. |
| `lookback_list` | `list[int]` | Historical windows to use in reconstruction/inference. |
| `forecast_horizon_list` | `list[int]` | Forecast horizons requested by the client. |
| `head_size` | `int` | Transformer attention head dimension. |
| `num_heads` | `int` | Number of attention heads. |
| `ff_dim` | `int` | Feed-forward network hidden dimension. |
| `num_transformer_blocks` | `int` | Number of Transformer encoder blocks. |
| `mlp_units` | `int` | MLP layer width. |
| `activation_function` | `str` | MLP activation function name (example: `relu`). |
| `cutoff_date` | `str` | Data cutoff date used by the pipeline. |
| `data_path` | `str` | Source data path. |
| `save_path` | `str` | Output path for predictions/artifacts. |

### Optional fields with defaults

| Field | Type | Default | Description |
|---|---|---|---|
| `dropout` | `float` | `0.0` | Dropout rate for regularization. |
| `learning_rate` | `float` | `0.001` | Optimizer learning rate. |
| `epochs` | `int` | `50` | Number of training epochs. |
| `batch_size` | `int` | `32` | Batch size. |
| `covid_token` | `bool` | `True` | Include COVID period token feature. |
| `positional_encoding` | `bool` | `True` | Enable positional encoding. |
| `evaluate_model` | `bool` | `True` | Run evaluation after model reconstruction. |

Example payload:

```json
{
  "code": "J00",
  "lookback_list": [30],
  "forecast_horizon_list": [7, 30],
  "head_size": 128,
  "num_heads": 4,
  "ff_dim": 512,
  "num_transformer_blocks": 4,
  "mlp_units": 128,
  "activation_function": "relu",
  "dropout": 0.1,
  "learning_rate": 0.001,
  "epochs": 50,
  "batch_size": 32,
  "cutoff_date": "2025-09-30",
  "covid_token": true,
  "positional_encoding": true,
  "evaluate_model": true,
  "data_path": "../data/FINAL_DB/full_CAT1.parquet",
  "save_path": "../production_predictions/final_output_predictions"
}
```

## Validation and Error Semantics

FastAPI uses Pydantic model validation before route handlers execute.

- Invalid request body shape/type returns HTTP `422`.
- Missing required fields returns HTTP `422`.
- Route-level runtime errors are returned as HTTP exceptions (`404`, `500`, `503`) depending on the endpoint.

## Contract Evolution Guidance

To keep API compatibility stable:

- Additive changes: prefer adding optional fields with safe defaults.
- Breaking changes: version endpoint or request schema explicitly.
- Document every field change in release notes and external report documents.
