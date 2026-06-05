# Predap API External Integration Report

Document version: 1.0  
Last updated: 2026-04-21  
Audience: External partner engineering teams integrating with Predap services

## 1. Executive Summary

The Predap API provides production inference and data lifecycle operations for healthcare demand forecasting.

Core capabilities:

- Add a new data point to the forecasting dataset.
- Launch asynchronous model reconstruction and prediction jobs.
- Poll background job status and retrieve job outcomes.
- Clean outdated rows from prediction storage.

The API is implemented with FastAPI and Pydantic. This guarantees strict request validation and machine-readable OpenAPI contracts.

## 2. API Stack and Contract Sources

- Web framework: FastAPI
- Validation and schema typing: Pydantic v2
- API contract endpoints:
  - Swagger: `/docs`
  - ReDoc: `/redoc`
  - OpenAPI JSON: `/openapi.json`

## 3. Base URL and Routing

Environment-specific base URL is deployment-defined.

Endpoint groups:

- Root health endpoint: `/`
- Production namespace: `/production/*`

## 4. Endpoint Catalogue

### 4.1 GET /

Purpose: Simple API availability verification.

Success response:

```json
{
  "message": "Welcome, the Predap API is running!"
}
```

### 4.2 POST /production/add_new_data

Purpose: Append a new data row and persist updated dataset.

Input contract:

- Optional path/date/config flags.
- Optional explicit value vector (`provided_data`). If data is not provided, the mean of the last 3 years is automatically calculated as the value vector to add.
- Server can fallback to internal defaults where request fields are omitted.

Output contract:

- Operation status and message.
- Persisted file path.
- Total row count.
- New row content summary.

Failure codes:

- `404` source file not found.
- `500` internal processing failure.

### 4.3 POST /production/model_reconstruction_pipeline

Purpose: Trigger model reconstruction and prediction in background mode.

Input contract:

- Diagnostic/service code.
- Lookback windows and forecast horizons.
- Transformer and optimization hyperparameters.
- Data and output paths.

Output contract (`202 Accepted`):

- `job_id`
- `status` (`queued`)
- `status_endpoint` for polling.

Failure codes:

- `422` validation failure.
- `503` Redis unavailable.

### 4.4 GET /production/model_reconstruction_pipeline/{job_id}

Purpose: Poll async job execution status.

Job lifecycle states:

- `queued`
- `running`
- `succeeded`
- `failed`

When successful, response includes:

- Output row count.
- Output path.

Failure codes:

- `404` unknown/expired job.
- `503` Redis unavailable.

### 4.5 DELETE /production/delete_old_data

Purpose: Remove outdated forecast rows from production prediction storage.

Output contract:

- Success status and updated dataset path.

Failure codes:

- `500` internal processing failure.

## 5. Pydantic Data Contracts

### 5.1 AddNewDataRequest

Primary fields:

- `new_data_path`, `cutoff_date`, `max_date`
- `eliminate_covid_data`, `covid_token`
- `provided_data`
- `save_path`, `delete_old`

### 5.2 ModelReconstructionRequest

Primary fields:

- Model target and sequence definitions: `code`, `lookback_list`, `forecast_horizon_list`
- Transformer configuration: `head_size`, `num_heads`, `ff_dim`, `num_transformer_blocks`, `mlp_units`
- Training setup: `activation_function`, `dropout`, `learning_rate`, `epochs`, `batch_size`
- Data/runtime flags: `cutoff_date`, `covid_token`, `positional_encoding`, `evaluate_model`
- Paths: `data_path`, `save_path`

Validation behavior:

- Missing required fields and type mismatches are rejected with `422`.

## 6. Async Job Management Details

The background reconstruction endpoint uses Redis to persist job metadata.

Operational parameters:

- `REDIS_URL` default: `redis://redis:6379/0`
- `JOB_TTL_SECONDS` default: `86400`

Recommendations for client applications:

- Treat `job_id` as an opaque identifier.
- Poll status endpoint with retry/backoff.
- Consider timeout and dead-letter handling for `failed` jobs.

## 7. Integration Checklist for External Teams

1. Confirm API base URL and environment access.
2. Validate connectivity with `GET /`.
3. Use OpenAPI JSON to generate client SDKs if desired.
4. Implement request validation client-side mirroring Pydantic field types.
5. For reconstruction requests, implement polling workflow on `job_id`.
6. Capture and log all non-2xx responses for support escalation.

## 8. Security and Reliability Notes

- Authentication/authorization strategy is deployment-specific and should be agreed with platform owners.
- Ensure transport security using HTTPS at ingress.
- Redis availability is required for async reconstruction operations.
- Use idempotency controls at client side where duplicate submissions are possible.

## 9. Support and Escalation

Provide the following when raising incidents:

- Timestamp and environment.
- Endpoint and HTTP status.
- Request payload (masked for sensitive fields).
- Response body and `job_id` when applicable.

## 10. Change Control

This report reflects the current FastAPI router and Pydantic schema implementation as of the date above.
Future changes should be versioned and communicated with updated OpenAPI and report artifacts.
