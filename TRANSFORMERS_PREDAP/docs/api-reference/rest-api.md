# REST API (FastAPI)

Predap exposes a production-facing REST API built with FastAPI and typed request models powered by Pydantic.

## API Surface

- Application entrypoint: `api/main.py`
- Router namespace: `api/routers/production.py`
- Pydantic schemas: `api/schemas/production_schemas.py`
- Prefix for production endpoints: `/production`

## Run and Inspect

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive contracts are available at runtime:

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- OpenAPI spec: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

## Endpoint Reference

### GET /

Health-check style root endpoint.

- Handler: `api/main.py` -> `root`
- Success response:

```json
{
  "message": "Welcome, the Predap API is running!"
}
```

### POST /production/add_new_data

Appends one new row to the production dataset.

Behavior:

- If `provided_data` is omitted, the pipeline imputes values using a seasonal strategy.
- Optional request fields fall back to values from `BaseTransformerConfig`.
- Persisted output is written to `save_path`.

Request model: `AddNewDataRequest`.

Sample request:

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

Sample success response:

```json
{
  "status": "success",
  "message": "New data added successfully!",
  "saved_path": "../data/FINAL_DB1/full_CAT1.parquet",
  "total_rows": 6205,
  "new_row": {
    "timestamp": "2025-10-01 00:00:00",
    "J00": 45.0
  }
}
```

Error responses:

- `404`: source data file not found.
- `500`: pipeline or persistence failure.

### POST /production/model_reconstruction_pipeline

Starts model reconstruction and prediction as an asynchronous background job.

Key points:

- Returns `202 Accepted` immediately.
- Creates a Redis-backed job record with a generated `job_id`.
- Executes reconstruction in background via FastAPI `BackgroundTasks`.
- Poll job status using `GET /production/model_reconstruction_pipeline/{job_id}`.

Request model: `ModelReconstructionRequest`.

Sample request:

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

Sample `202` response:

```json
{
  "status": "queued",
  "job_id": "4f1ec6f5-3f6f-4fd4-9f9d-7da9dcdbed13",
  "status_endpoint": "/production/model_reconstruction_pipeline/4f1ec6f5-3f6f-4fd4-9f9d-7da9dcdbed13",
  "message": "Model reconstruction started in the background."
}
```

Potential errors:

- `422`: invalid request body (Pydantic validation failure).
- `503`: Redis unavailable while creating job.

### GET /production/model_reconstruction_pipeline/{job_id}

Returns job status and (if available) result payload.

Job statuses:

- `queued`
- `running`
- `succeeded`
- `failed`

Sample succeeded response:

```json
{
  "job_id": "4f1ec6f5-3f6f-4fd4-9f9d-7da9dcdbed13",
  "status": "succeeded",
  "created_at": "2026-04-21T09:34:55.402631+00:00",
  "updated_at": "2026-04-21T09:36:22.778281+00:00",
  "finished_at": "2026-04-21T09:36:22.777902+00:00",
  "error": null,
  "result": {
    "rows": 1240,
    "output_path": "../production_predictions/final_output_predictions"
  }
}
```

Potential errors:

- `404`: unknown or expired job id.
- `503`: Redis unavailable while reading status.

### DELETE /production/delete_old_data

Runs cleanup of outdated forecast rows in the production predictions dataset.

Sample success response:

```json
{
  "status": "success",
  "message": "Old data deleted successfully from: ../production_predictions/final_output_predictions.parquet",
  "updated_dataset_path": "../production_predictions/final_output_predictions.parquet"
}
```

Potential errors:

- `500`: cleanup pipeline failure.

## Pydantic Validation Model Reference

Detailed field-by-field schema documentation is in [Pydantic Models](pydantic-models.md).

## Operational Notes

- Redis URL comes from `REDIS_URL` (default: `redis://redis:6379/0`).
- Job metadata TTL comes from `JOB_TTL_SECONDS` (default: `86400`).
- The model reconstruction endpoint is asynchronous by design; clients should poll the status endpoint.

## External Distribution

A client-facing report is available in markdown and PDF:

- Markdown: [External API Report](external-api-report.md)
- PDF: [Predap API External Report (PDF)](Predap_API_External_Report.pdf)
