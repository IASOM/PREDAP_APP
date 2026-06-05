<div class="rumia-hero" markdown="1">

<p class="rumia-eyebrow">RUMIA HealthTech Documentation</p>

# Predap — Healthcare Demand Forecasting

Predap is a deep learning ecosystem for forecasting diagnostic visit demand in healthcare systems. It uses a three-phase Transformer-based residual correction pipeline with production-ready inference capabilities via FastAPI.

<div class="rumia-cta-row">

[Installation](getting-started/installation.md){ .md-button .md-button--primary }
[Quickstart](getting-started/quickstart.md){ .md-button }
[API Reference](api-reference/rest-api.md){ .md-button }

</div>

</div>

## What Predap Delivers

<div class="rumia-grid" markdown="1">

<div class="rumia-card" markdown="1">

### Reliable Forecasting

High-accuracy multi-horizon predictions with a transparent residual correction pipeline.

- 1 to 365 days ahead
- Structured experiment tracking
- Production-ready inference

</div>


<div class="rumia-card" markdown="1">

### Designed for Teams

Clear navigation, calm layouts, and documentation patterns that support technical reading.

- FastAPI deployment
- Hydra experiments
- MkDocs Material base

</div>

</div>

## Architecture Overview

```mermaid
graph TD
    subgraph Data Ingestion
        A[Raw Parquet Data] --> B[data_preparation.py]
        B --> C[Temporal Feature Engineering]
        C --> D[Cyclical Encodings + Holiday/Vacation Flags]
    end

    subgraph Phase 1 [Phase 1 - Univariate Transformer]
        D --> E[RevIN Normalization]
        E --> F[Positional Encoding]
        F --> G[Multi-Head Self-Attention x N Blocks]
        G --> H[AveragePooling1D + MLP Head]
        H --> I[Baseline Forecast y_1]
    end

    subgraph Phase 2 [Phase 2 - Diagnostic Residual Transformer]
        I --> J[Compute Residuals: r_1 = y - y_1]
        K[Diagnostic Covariates LMLR + GCausal] --> L
        J --> L[Hybrid LSTM-Transformer]
        L --> M[Predicted Residuals r_hat_1]
        M --> N[Corrected Forecast y_2 = y_1 + r_hat_1]
    end

    subgraph Phase 3 [Phase 3 - Seasonal Residual Transformer]
        N --> O[Compute Residuals: r_2 = y - y_2]
        P[Seasonal Covariates DoW / Month / Holidays] --> Q
        O --> Q[Hybrid LSTM-Transformer]
        Q --> R[Predicted Residuals r_hat_2]
        R --> S[Final Forecast y_3 = y_2 + r_hat_2]
    end

    subgraph Production
        S --> T[Model Quantization float16]
        T --> U[FastAPI Inference Endpoint]
        U --> V[Client Applications]
    end

    
```

## Experiment Tracking

```mermaid
graph TD
    subgraph Experiment Tracking
        W[MLflow] --> X[Metrics + Artifacts + Models]
        Y[Hydra Grid Search] --> W
    end
```



## Three-Phase Pipeline Summary

| Phase | Pipeline Class | Input | Output |
|-------|---------------|-------|--------|
| **1. Univariate** | `UnivariateTransformerPipeline` | Target time series + temporal features | Baseline forecast $\hat{y}_1$ |
| **2. Diagnostic Residual** | `DiagnosticResidualTransformerPipeline` | Residuals $r_1$ + diagnostic covariates | Corrected forecast $\hat{y}_2 = \hat{y}_1 + \hat{r}_1$ |
| **3. Seasonal Residual** | `SeasonalResidualTransformerPipeline` | Residuals $r_2$ + seasonal covariates | Final forecast $\hat{y}_3 = \hat{y}_2 + \hat{r}_2$ |

## Key Capabilities

- **Accurate multi-horizon forecasts** of healthcare diagnostic visits from 1 to 365 days ahead
- **Three-phase residual correction** that progressively refines predictions from baseline to seasonal enrichment
- **Production-ready deployment** with quantized models, FastAPI endpoints, and automated data ingestion
- **Experiment tracking** with MLflow and Hydra-based grid search
- **COVID-aware logic** built around pandemic wave handling and the Catalan public health calendar

## Quick Links

[Installation](getting-started/installation.md){ .md-button .md-button--primary }
[5-Minute Quickstart](getting-started/quickstart.md){ .md-button }
[API Reference](api-reference/rest-api.md){ .md-button }
