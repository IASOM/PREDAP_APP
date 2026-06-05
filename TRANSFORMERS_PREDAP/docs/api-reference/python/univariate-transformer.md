# Univariate Transformer API

## Model Architecture

::: univariate_transformer.model_architecture_univ_transformer
    options:
      show_root_heading: true
      show_source: true
      members:
        - build_model
        - RevIN
        - PositionalEncoding
        - CustomCosineDecay

---

## Alternative Architectures

### Base Transformer (Default)

::: univariate_transformer.transformer_architechtures.model_architechture_base_tranformer
    options:
      show_root_heading: true
      members:
        - build_base_model

### Informer

::: univariate_transformer.transformer_architechtures.model_architechture_informer
    options:
      show_root_heading: true
      members:
        - build_informer_model

### LogSparse Transformer

::: univariate_transformer.transformer_architechtures.model_architechture_log_transformer
    options:
      show_root_heading: true
      members:
        - build_log_transformer_model

### LSTNet

::: univariate_transformer.transformer_architechtures.model_architechture_LSTNet
    options:
      show_root_heading: true
      members:
        - build_lstnet_model

---

## Training & Evaluation

::: univariate_transformer.training_evaluation_univ_transformer
    options:
      show_root_heading: true
      members:
        - train_given_model_and_data
        - evaluate_model_sliding_window
        - evaluate_model_basic

---

## Evaluation

::: univariate_transformer.evaluation_univ_transformer
    options:
      show_root_heading: true
      members:
        - evaluate_univ_transformer

---

## Visualization

::: univariate_transformer.visualization_univ_transformer
    options:
      show_root_heading: true
      members:
        - plt_model
        - plot_predictions_with_waves
        - plot_training_history

---

## Utilities

::: univariate_transformer.utils_univ_transformer
    options:
      show_root_heading: true
      members:
        - extract_model_params
        - load_and_evaluate_models
        - create_pandemic_waves_df
        - setup_gpu_memory
        - load_mlflow_model_history
