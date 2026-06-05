# Residual Transformers API

The residual multivariate transformer module (`src/residual_multivariate_transformers/`) implements Phases 2 and 3 of the Predap pipeline — diagnostic and seasonal residual correction.

---

## Model Architecture

::: residual_multivariate_transformers.model_architecture_residual_transformer
    options:
      show_root_heading: true
      show_source: true
      members:
        - hybrid_lstm_transformer_model
        - transformer_encoder

---

## Training & Evaluation

::: residual_multivariate_transformers.training_evaluation_residual_transformer
    options:
      show_root_heading: true
      members:
        - train_given_model_and_data
        - evaluate_model
        - load_trained_model
        - get_callbacks

---

## Data Utilities

::: residual_multivariate_transformers.utils_residual_transformer
    options:
      show_root_heading: true
      members:
        - split_train_test
        - filter_diagnostics_covariates
        - learn_covariates
        - prepare_residual_data
        - prepare_base_model_data
        - load_base_model_transformer
        - validate_data_shapes

---

## Visualization

::: residual_multivariate_transformers.visualization_residual_transformer
    options:
      show_root_heading: true
      members:
        - plot_residuals_analysis
        - plot_stepwise_errors_comparison
        - plot_predictions_with_pandemic_waves
        - save_performance_results
        - compare_model_performance
