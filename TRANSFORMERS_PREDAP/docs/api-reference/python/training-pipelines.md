# Training Pipelines API

The pipeline classes provide the high-level orchestration for the three-phase training loop. Each pipeline encapsulates configuration, data preparation, model building, training, and evaluation into a single `run_complete_pipeline()` call.

---

## Phase 1: Univariate Transformer Pipeline

::: main_train_univ_transformer_class.TransformerUnivConfig
    options:
      show_root_heading: true
      show_source: false

::: main_train_univ_transformer_class.UnivariateTransformerPipeline
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - setup_environment
        - load_diagnostic_covariates
        - prepare_data
        - build_model
        - setup_callbacks
        - compile_model
        - train_model
        - evaluate_model
        - run_complete_pipeline
        - get_results_summary

---

## Phase 2: Diagnostic Residual Pipeline

::: main_train_diagnostic_residual_transformer_class.DiagnosticResidualTransformerConfig
    options:
      show_root_heading: true
      show_source: false

::: main_train_diagnostic_residual_transformer_class.DiagnosticResidualTransformerPipeline
    options:
      show_root_heading: true
      show_source: true
      members:
        - prepare_base_model_data
        - prepare_covariate_data
        - build_residual_model
        - train_residual_model
        - evaluate_residual_model
        - generate_visualizations
        - run_complete_pipeline

---

## Phase 3: Seasonal Residual Pipeline

::: main_train_seasonal_residual_transformer_class.SeasonalResidualTransformerConfig
    options:
      show_root_heading: true
      show_source: false

::: main_train_seasonal_residual_transformer_class.SeasonalResidualTransformerPipeline
    options:
      show_root_heading: true
      show_source: true

---

## Evaluation Pipeline

::: main_eval_models.ConfigEvalTransformers
    options:
      show_root_heading: true
      show_source: false

::: main_eval_models.EvalTransformers
    options:
      show_root_heading: true
      show_source: true
      members:
        - setup_classes
        - compute_predictions
        - compute_diagnostic_predictions
        - compute_seasonal_predictions
        - run_eval_x_step
        - compare_eval_steps

---

## Backward-Compatible Function API

::: main_train_univ_transformer_class.main_univ_transformer
    options:
      show_root_heading: true

::: main_train_univ_transformer_class.run_batch_experiments
    options:
      show_root_heading: true
