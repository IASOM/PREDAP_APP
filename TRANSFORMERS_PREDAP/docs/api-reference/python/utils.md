# Utilities API

---

## Experiment Utilities

::: utils.experiments_utils
    options:
      show_root_heading: true
      show_source: true
      members:
        - cleanup_ram
        - memory_cleanup
        - smart_read
        - safe_float
        - initialize_results_tracking
        - load_json_codes_list
        - compute_dynamic_batch_size

---

## Evaluation & Plotting Utilities

::: utils.evaluation_plot_utils
    options:
      show_root_heading: true
      show_source: true
      members:
        - plot_predictions_with_waves
        - crps
        - smape
        - pinball_loss
        - mean_absolute_percentage_error
        - plot_stepwise_errors
        - plot_errors_over_time_with_waves
        - evaluate_error_significance_pandemic_waves

---

## Configuration

::: config.base_transformer_config.BaseTransformerConfig
    options:
      show_root_heading: true
      show_source: true
      members:
        - __post_init__
        - to_dict
        - print_config
        - get_model_name
        - get_diagnostic_residual_model_name
        - get_seasonal_residual_model_name
        - get_lr_schedule_params
