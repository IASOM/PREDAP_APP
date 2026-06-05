# Evaluation & Metrics

Predap evaluates forecasting performance using a comprehensive suite of metrics, applied on the **original (un-normalized) scale** after inverse transformation. The evaluation framework is implemented in `src/utils/evaluation_plot_utils.py` and `src/univariate_transformer/evaluation_univ_transformer.py`.
Only the MAE, RMSE, MSE and WAPE metrics are loaded to Mlflow. 

---

## Core Metrics

### Mean Squared Error (MSE)

$$
\mathrm{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

### Root Mean Squared Error (RMSE)

$$
\mathrm{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

### Mean Absolute Error (MAE)

$$
\mathrm{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

### Weighted Absolute Percentage Error (WAPE)

$$
\mathrm{WAPE} = \frac{\sum_{i=1}^{n} |y_i - \hat{y}_i|}{\sum_{i=1}^{n} |y_i|}
$$

### Symmetric Mean Absolute Percentage Error (sMAPE)

$$
\mathrm{sMAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}
$$

### Continuous Ranked Probability Score (CRPS)

For deterministic forecasts, the CRPS reduces to the MAE:

$$
\mathrm{CRPS} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

### Pinball Loss (Quantile Loss)

At quantile $\tau$ (default $\tau = 0.5$ for the median):

$$
L_\tau(y, \hat{y}) = \begin{cases}
\tau \cdot (y - \hat{y}) & \text{if } y \geq \hat{y} \\
(1 - \tau) \cdot (\hat{y} - y) & \text{if } y < \hat{y}
\end{cases}
$$

### Mean Absolute Percentage Error (MAPE)

$$
\mathrm{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i + \epsilon}\right|
$$

!!! warning "Near-Zero Protection"
    A configurable `epsilon` is added to the denominator to prevent division by zero. A `warn_threshold` flags codes where many true values are near zero, making MAPE unreliable.

---

## MLflow Metric Logging

During training, the previously specified metrics of each transformer phase are logged to MLflow under structured namespaces:

| Metric Key | Phase | Description |
|-----------|-------|-------------|
| `eval/univ_transformer_loss` | 1 | Univariate test loss |
| `eval/univ_transformer_mae` | 1 | Univariate MAE |
| `eval/univ_transformer_mse` | 1 | Univariate MSE |
| `eval/univ_transformer_rmse` | 1 | Univariate RMSE |
| `eval/univ_transformer_wape` | 1 | Univariate WAPE |
| `eval/residual_diagnostics_model_mae` | 2 | Diagnostic residual MAE |
| `eval/residual_diagnostics_model_mse` | 2 | Diagnostic residual MSE |
| `eval/residual_diagnostics_model_rmse` | 2 | Diagnostic residual RMSE |
| `eval/residual_diagnostics_model_wape` | 2 | Diagnostic residual WAPE |
| `eval/residual_seasonal_model_mae` | 3 | Seasonal residual MAE |
| `eval/residual_seasonal_model_mse` | 3 | Seasonal residual MSE |
| `eval/residual_seasonal_model_rmse` | 3 | Seasonal residual RMSE |
| `eval/residual_seasonal_model_wape` | 3 | Seasonal residual WAPE |
| `duration/phase_{n}_duration_seconds` | 1–3 | Per-phase training time |
| `total_training_duration_minutes` | All | End-to-end duration |

---

## Evaluation Modes

### Sliding Window Evaluation

The `evaluate_model_sliding_window()` function applies the model to overlapping test windows, computing per-step error metrics:

```python
from src.univariate_transformer.training_evaluation_univ_transformer import (
    evaluate_model_sliding_window
)

evaluate_model_sliding_window(
    model=model,
    model_name="J00_base_transformer_7fh_512ff_14lb",
    X_test=X_test,
    Y_test=Y_test,
    date_list=date_list,
    df_waves=pandemic_waves_df,
    sliding_window=True
)
```

### Original-Scale Evaluation

The full evaluation pipeline in `evaluate_univ_transformer()`:

1. Loads the saved `.keras` model
2. Prepares test data using `prepare_data(train=False)`
3. Generates predictions
4. **Inverse-transforms** predictions and actual values to the original scale
5. Computes MAE, MSE, RMSE, WAPE on the original scale
6. Generates visualization plots
7. Logs everything to MLflow

---

## Visualization Suite

### Predictions vs Actuals with Pandemic Waves

`plot_predictions_with_waves()` overlays actual vs predicted values on a timeline, with colored vertical bands indicating each of the five Catalan pandemic waves. This makes it easy to assess model degradation during unprecedented periods.

### Per-Step Error Analysis

`plot_stepwise_errors()` produces per-forecast-step breakdowns of:

- MSE per step (step 1, step 2, ..., step $h$)
- MAE per step
- Bias per step
- CRPS per step
- Pinball loss per step
- sMAPE per step

This reveals whether the model degrades at longer horizons.

### Residual Analysis (Phase 2 & 3)

`plot_residuals_analysis()` produces a 4-panel figure:

1. **Time series** — Original vs corrected predictions with uncertainty bands ($\pm 1.96 \sigma$)
2. **Residuals over time** — Plot of the residual error at each time step
3. **Scatter plot** — Predicted vs actual with $R^2$ line
4. **Error distribution** — Histogram of residuals with Gaussian fit

### Pandemic Wave Error Significance

`evaluate_error_significance_pandemic_waves()` performs statistical hypothesis testing:

- **t-test** and **Mann-Whitney U test** comparing errors inside vs outside each pandemic wave
- Reports p-values to determine if pandemic periods cause significant forecast degradation

---

## Multi-Horizon Evaluation

The `EvalTransformers` class in `src/main_eval_models.py` supports evaluating multiple `(lookback, forecast)` combinations:

```python
from src.main_eval_models import EvalTransformers, ConfigEvalTransformers

evaluator = EvalTransformers(ConfigEvalTransformers())
results = evaluator.run_eval_x_step(
    lookback_list=[7, 14, 60],
    forecast_list=[7, 14, 30],
    code="J00",
)
evaluator.compare_eval_steps(results, found_lookbacks, max_shape)
```

This generates comparison plots across horizons, identifying which `(lookback, forecast)` pair yields the best performance for each diagnostic code.
