import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def crps(y_true, y_pred):
    """
    Computes Continuous Ranked Probability Score (CRPS).
    This version assumes deterministic predictions (for probabilistic forecasting, use predictive distributions).
    """
    return np.mean((y_true - y_pred) ** 2)
    
def smape(y_true, y_pred):
    """
    Computes Symmetric Mean Absolute Percentage Error (SMAPE).
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred)
    smape_val = np.mean(np.where(denominator == 0, 0, diff / denominator)) * 100
    return smape_val

def pinball_loss(y_true, y_pred, tau=0.5):
    """
    Computes Pinball Loss for a given quantile tau (default = 0.5 for median forecast).
    """
    error = y_true - y_pred
    return np.mean(np.maximum(tau * error, (tau - 1) * error))

def mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-8, warn_threshold=0.1):
    """
    Computes MAPE while avoiding division by near-zero actual values.
    
    Args:
        y_true (array): Actual target values.
        y_pred (array): Predicted target values.
        epsilon (float): Minimum threshold for |y_true| to be included in MAPE.
        warn_threshold (float): Warn if too much data is excluded (default 10%).
    
    Returns:
        float: Mean Absolute Percentage Error (MAPE) in percent.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Mask small or zero actuals
    mask = np.abs(y_true) > epsilon

    # Warn if too many values are excluded
    if np.mean(~mask) > warn_threshold:
        print(f"⚠️ Warning: {np.mean(~mask) * 100:.1f}% of values excluded from MAPE due to small actuals (< {epsilon})")

    # Compute MAPE only on valid values
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
