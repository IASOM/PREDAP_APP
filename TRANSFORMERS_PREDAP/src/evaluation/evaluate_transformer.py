
import os
import json
import sys
import tensorflow as tf
import pandas as pd
from datetime import datetime
import numpy as np

from training.training_residual_transformer import load_trained_model
from utils.mlflow_logger import MLflowLogger

def save_performance_results(model_name, loss, 
                             original_mae, original_mse, 
                             original_rmse, original_wape,
                             corrected_mae, corrected_mse, 
                             corrected_rmse, corrected_wape,
                             forecast, lookback, code, output_dir="results"):
    """
    Save performance comparison results to a JSON file.
    
    Parameters:
    -----------
    model_name : str
        Name of the residual model being evaluated
    loss : float
        Loss value for the model
    original_mae, original_mse, original_rmse : float
        Performance metrics for the original base model
    corrected_mae, corrected_mse, corrected_rmse : float
        Performance metrics for the residual-corrected model
    forecast : int
        Forecast horizon used
    lookback : int
        Lookback window used
    code : str
        Target diagnostic code
    output_dir : str
        Directory to save the results file
    
    Returns:
    --------
    str
        Path to the saved JSON file
    """
    
    # Create output directory if it doesn't exist
    if not os.path.exists('../' +output_dir):
        os.makedirs('../' + output_dir)
        print(f"Created directory: {output_dir}")
    
    # Calculate improvements
    mae_improvement = float((original_mae - corrected_mae) / original_mae * 100) if original_mae != 0 else 0
    mse_improvement = float((original_mse - corrected_mse) / original_mse * 100) if original_mse != 0 else 0
    rmse_improvement =float((original_rmse - corrected_rmse) / original_rmse * 100) if original_rmse != 0 else 0
    wape_improvement =float((original_wape - corrected_wape) / original_wape * 100) if original_wape != 0 else 0
    
    # Create results dictionary
    results = {
        "model_info": {
            "residual_model_name": model_name,
            "target_code": code,
            "forecast_horizon": forecast,
            "lookback_window": lookback,
            "evaluation_timestamp": datetime.now().isoformat(),
            "model_type": "Residual Multivariate Transformer"
        },
        "original_model_performance": {
            "MAE": round(float(original_mae), 6),
            "MSE": round(float(original_mse), 6),
            "RMSE": round(float(original_rmse), 6),
            "WAPE": round(float(original_wape), 6)
        },
        "corrected_model_performance": {
            "MAE": round(float(corrected_mae), 6),
            "MSE": round(float(corrected_mse), 6),
            "RMSE": round(float(corrected_rmse), 6),
            "WAPE": round(float(corrected_wape), 6)

        },
        "improvements": {
            "MAE_improvement_percent": round(float(mae_improvement), 2),
            "MSE_improvement_percent": round(float(mse_improvement), 2),
            "RMSE_improvement_percent": round(float(rmse_improvement), 2),
            "WAPE_improvement_percent": round(float(wape_improvement), 2),
            "overall_assessment": "positive" if float(mae_improvement) > 0 else "negative"
        },
        "summary": {
            "best_metric": "MAE" if abs(mae_improvement) >= max(abs(mse_improvement), abs(rmse_improvement)) else 
                         "MSE" if abs(mse_improvement) >= abs(rmse_improvement) else "RMSE",
            "best_improvement": max(mae_improvement, mse_improvement, rmse_improvement),
            "average_improvement": round((mae_improvement + mse_improvement + rmse_improvement) / 3, 2)
        }
    }

    MLflowLogger(active=True).log_metrics({
        f"eval/{model_name}_loss": loss,
        f"eval/{model_name}_mae": corrected_mae,
        f"eval/{model_name}_mse": corrected_mse,
        f"eval/{model_name}_rmse": corrected_rmse,
        f"eval/{model_name}_wape": corrected_wape
    })
    
    # Generate filename based on model name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_model_name = model_name.replace('.keras', '').replace('/', '_').replace('\\', '_')
    filename = f"performance_{clean_model_name}_{timestamp}.json"
    filepath = os.path.join('../' +output_dir, filename)
    
    # Save to JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\n📊 Performance results saved to: {filepath}")
    
    return filepath


def save_univ_performance_results(model_name, loss, 
                             original_mae, original_mse, 
                             original_rmse, original_wape,
                             forecast, lookback, code, output_dir="results"):
    """
    Save performance comparison results to a JSON file.
    
    Parameters:
    -----------
    model_name : str
        Name of the residual model being evaluated
    loss : float
        Loss value for the model
    original_mae, original_mse, original_rmse : float
        Performance metrics for the original base model
    corrected_mae, corrected_mse, corrected_rmse : float
        Performance metrics for the residual-corrected model
    forecast : int
        Forecast horizon used
    lookback : int
        Lookback window used
    code : str
        Target diagnostic code
    output_dir : str
        Directory to save the results file
    
    Returns:
    --------
    str
        Path to the saved JSON file
    """
    
    # Create output directory if it doesn't exist
    if not os.path.exists('../' +output_dir):
        os.makedirs('../' + output_dir)
        print(f"Created directory: {output_dir}")
    
    # Create results dictionary
    results = {
        "model_info": {
            "univ_model_name": model_name,
            "target_code": code,
            "forecast_horizon": forecast,
            "lookback_window": lookback,
            "evaluation_timestamp": datetime.now().isoformat(),
            "model_type": "Residual Multivariate Transformer"
        },
        "univ_model_performance": {
            "MAE": round(float(original_mae), 6),
            "MSE": round(float(original_mse), 6),
            "RMSE": round(float(original_rmse), 6),
            "WAPE": round(float(original_wape), 6)
        },
        
    }

    MLflowLogger(active=True).log_metrics({
        f"eval/{model_name}_loss": loss,
        f"eval/{model_name}_mae": original_mae,
        f"eval/{model_name}_mse": original_mse,
        f"eval/{model_name}_rmse": original_rmse,
        f"eval/{model_name}_wape": original_wape
    })
    
    # Generate filename based on model name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_model_name = model_name.replace('.keras', '').replace('/', '_').replace('\\', '_')
    filename = f"performance_{clean_model_name}_{timestamp}.json"
    filepath = os.path.join('../' +output_dir, filename)
    
    # Save to JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\n📊 Performance results saved to: {filepath}")
    
    return filepath
