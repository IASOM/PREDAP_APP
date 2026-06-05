# src — Module Overview

This document describes the `src` package layout, the main modules in it, and how to use their key functions. It is intended to help developers quickly understand where to find preprocessing, training, and utility code used across the project.

**Overview**

- The `src` folder contains the project's core implementation: data preprocessing, model architecture, training routines, evaluation helpers, and visualization utilities.

**Top-level folders**

- `config/`: Configuration helpers and constants used by training and inference.
- `data_utils/`: Data ingestion and preprocessing utilities (detailed below).
- `evaluation/`: Evaluation metrics and result-aggregation utilities.
- `model_architechture/`: Model definitions and components used to construct the transformers/residual models.
- `training/`: Training loops, checkpointing and scheduling code.
- `utils/`: Misc helper functions used across modules (I/O, logging, small helpers).
- `visualization_func/`: Plotting utilities for analysis and reporting.

**`data_utils` (detailed)**

Files:

- `data_preparation.py` — routines to load raw CSVs, align time series and create train/val/test splits.
- `residual_data_preparation.py` — helpers to prepare residual series used by the residual-correction transformer variants.
- `normalizer.py` — scaling/normalization utilities (fit/transform wrappers for common scalers).
- `features.py` — time-series feature-generation utilities. (Primary functions documented below.)

Key functions in `features.py` and usage

- `eliminate_covid_dates(df: pd.DataFrame, covid_periods: list) -> pd.DataFrame`:
  - Purpose: Remove rows whose `timestamp` falls within any of the given COVID periods (list of `(start, end)` strings or timestamps).
  - Use: Pre-filter historical data to exclude anomalous pandemic periods before modeling.

- `add_covid_token(df: pd.DataFrame) -> pd.DataFrame`:
  - Purpose: Add a `covid_token` column (0/1) marking rows that fall within several pre-coded COVID waves.
  - Use: Provide a binary feature that models can use to account for pandemic effects.

- `cut_dataframe(df: pd.DataFrame, date_cutoff: str = "2010-01-01", max_date: str = '2026-12-31') -> pd.DataFrame`:
  - Purpose: Filter the DataFrame to a date window between `date_cutoff` and `max_date` (inclusive).
  - Use: Ensure datasets are bounded to years of interest before downstream processing.

- `prepare_time_series_features(df: pd.DataFrame, categorical_vars, cutoff_date='2010-01-01', max_date='2027-09-30', scaler=None, eliminate_covid_data=False, covid_dates=None) -> pd.DataFrame`:
  - Purpose: Build a daily feature DataFrame covering the timestamp span in `df` with the following:
    - Cyclical encodings: `dow_sin`, `dow_cos`, `month_sin`, `month_cos`, `doy_sin`, `doy_cos`.
    - Boolean features: `Is_Weekend`.
    - Holiday indicator: computes a set of fixed and movable public holidays (including Good Friday and Easter Monday) and marks `Holiday` vs `No_Holiday`.
    - School vacation indicator: marks typical vacation periods as `Vacation` vs `No_Vacation`.
    - Optionally removes COVID periods and can accept categorical variables to one-hot encode.
  - Inputs:
    - `df`: input DataFrame with a `timestamp` column (or convertible to datetime).
    - `categorical_vars`: list of categorical columns names to one-hot encode on the date index (e.g., region-level flags).
    - `cutoff_date`, `max_date`: bounds for the generated date range.
    - `scaler`: reserved parameter (not currently applied inside the function but accepted for compatibility).
    - `eliminate_covid_data`, `covid_dates`: if `True`, removes rows in COVID periods; `covid_dates` must be provided.
  - Output: a DataFrame indexed by `timestamp` with the generated numeric and one-hot encoded features ready to merge with observed series.

Example usage

```python
from src.data_utils.features import prepare_time_series_features, add_covid_token

# df must contain a 'timestamp' column (datetime or parseable string)
feature_df = prepare_time_series_features(
    df, 
    categorical_vars=['Region', 'School_Type'],
    cutoff_date='2015-01-01',
    max_date='2026-12-31',
    eliminate_covid_data=True,
    covid_dates=[('2020-03-01','2020-06-30'), ('2020-10-01','2020-12-31')]
)

# Merge back with original series on 'timestamp' before modeling:
# merged = original_df.merge(feature_df, on='timestamp', how='left')
```

Guidelines and notes

- All functions expect a `timestamp` column in the input DataFrame; they convert it using `pd.to_datetime(..., errors='coerce')`.
- `prepare_time_series_features` builds daily features; align frequency before merging (e.g., resample original series to daily if needed).
- The holiday and school-vacation lists are examples tuned for the project's region; review and adapt if applying to other locales.

Where to go next

- See `src/data_utils/data_preparation.py` for how raw CSVs are loaded and how these feature functions are invoked in the pipeline.
- If you want more detailed inline docs for other modules, say which module and I will expand the README.

