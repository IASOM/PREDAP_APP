# Roadmap

Development roadmap for the **Predap** Transformer-based healthcare demand forecasting platform.

---

## Current Release — v1.0

- [x] Three-phase residual correction pipeline (Base → Diagnostic → Seasonal)
- [x] FastAPI inference endpoints with production pipelines
- [x] Hydra configuration for grid-search sweeps
- [x] MLflow experiment tracking with full metrics suite
- [x] Model quantisation (float16) for deployment
- [x] Reversible Instance Normalisation (RevIN) layer
- [x] Custom Cosine Decay learning rate scheduler
- [x] COVID-period handling (token & exclusion modes)
- [x] Attention-based multi-head Transformer architecture
- [x] Docker + Docker Compose deployment

---

## Short Term — v1.1

- [ ] Automated test suite (pytest) with CI/CD pipeline
- [ ] Multivariate Transformer variant (multi-feature input)
- [ ] Extended API documentation with Swagger examples
- [ ] Model versioning via MLflow Model Registry

---

## Medium Term — v1.2

- [ ] Support for exogenous covariates (weather, holidays, outbreaks)
- [ ] Patch-based tokenisation (PatchTST-inspired)
- [ ] Channel-independent decomposition (trend + seasonality heads)
- [ ] Warm-start fine-tuning from pre-trained checkpoints

---

## Long Term — v2.0

- [ ] Online / continual learning with incremental data updates
- [ ] Explainability module (attention-map visualisation, SHAP values)
- [ ] Multi-GPU distributed training (tf.distribute / Horovod)
- [ ] Event-driven pipeline with Airflow / Prefect orchestration

---

## Research Directions

| Area | Description | Status |
|------|-------------|--------|
| Temporal Fusion Transformer | Gate-based variable selection + multi-horizon attention | Exploring |
| Informer / Autoformer | Sparse attention for long-horizon forecasting | Planned |
| Diffusion-based forecasting | Score-based generative models for probabilistic outputs | Research |
| State Space Models | S4/Mamba-style recurrence for efficient long sequences | Research |
| Conformal Prediction | Distribution-free prediction intervals with coverage guarantees | Planned |

---

## Release History

| Version | Date | Highlights |
|---------|------|------------|
| v1.0 | 2025 | Three-phase pipeline, FastAPI, Hydra, MLflow, Docker |

---

!!! info "Contributing"
    Want to help build a feature? See the [Contributing Guide](contributing.md) for guidelines on submitting pull requests and opening issues.
