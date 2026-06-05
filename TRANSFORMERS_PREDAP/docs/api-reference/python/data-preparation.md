# Data Preparation API

The data preparation module (`src/utils/data_preparation.py`) is the backbone of all Predap data pipelines. It handles loading, filtering, normalization, feature engineering, and sequence creation.

---

## Main Pipeline Function

::: utils.data_preparation
    options:
      show_root_heading: true
      show_source: true
      members:
        - prepare_data
        - prepare_data_not_normalized

---

## Data Splitting & Filtering

::: utils.data_preparation
    options:
      show_root_heading: false
      members:
        - split_train_test
        - cut_dataframe
        - eliminate_covid_dates

---

## Normalization

::: utils.data_preparation
    options:
      show_root_heading: false
      members:
        - normalize_dataframe
        - inverse_transform_predictions

---

## Feature Engineering

::: utils.data_preparation
    options:
      show_root_heading: false
      members:
        - prepare_time_series_features
        - add_covid_token
        - prepare_time_series_covariates

---

## Sequence Generation

::: utils.data_preparation
    options:
      show_root_heading: false
      members:
        - generate_rolling_sequences_covariates
        - shift_covariates

---

## Utilities

::: utils.data_preparation
    options:
      show_root_heading: false
      members:
        - extract_dates
