try:
    import mlflow
except Exception:
    mlflow = None
import re
import os
import pandas as pd
import numpy as np
import tensorflow as tf


import mlflow
import pickle
import matplotlib.pyplot as plt


from config.config_manager import get_config
from src.data_utils.data_preparation import split_train_test


default_config = get_config()
PLOTS_DIR = default_config.plots_dir



def load_mlflow_model_history(model_name, model_type="univariate_transformer"):

    """
    Load training history from an MLflow Keras model.

    Args:
        model: MLflow Keras model
    Returns:
        dict: Training history  
    """
    raw_model_name = model_name.replace(".keras", "")
    history_path = f"../history/{raw_model_name}_history.pkl"

    if os.path.exists(history_path):
        print(f" Found saved history at: {history_path}")

        # Load the history
        with open(history_path, "rb") as f:
            history_data = pickle.load(f)
        
        # Convert to DataFrame for easier handling
        history_df = pd.DataFrame(history_data)
        history_df["epoch"] = range(1, len(history_df) + 1)

        
        # --- Log metrics ---
        for epoch, row in history_df.iterrows():
            for metric, value in row.items():
                if metric != "epoch":
                    from src.utils.mlflow_logger import MLflowLogger
                    MLflowLogger(active=True).log_metric(metric + "_" + model_type, float(value), step=int(row["epoch"]))
        
        # --- Create and log plots ---
        metric_groups = {
            "loss": ["loss", "val_loss"],
            "accuracy": ["accuracy", "val_accuracy"],
        }

        for group_name, keys in metric_groups.items():
            available = [k for k in keys if k in history_df.columns]
            if not available:
                continue

            plt.figure(figsize=(8, 4))
            for k in available:
                plt.plot(history_df["epoch"], history_df[k], label=k+ "_" + model_type, linewidth=2)
            plt.xlabel("Epoch")
            plt.ylabel(group_name.capitalize())
            plt.title(f"Training vs Validation {group_name.capitalize()}")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()

            plot_path = f"../{PLOTS_DIR}/{model_name}_{group_name}_curve.png"
            
            plt.close()
            os.makedirs('../' + PLOTS_DIR, exist_ok=True)
            plt.savefig(plot_path)
            # Log as artifact
            from src.utils.mlflow_logger import MLflowLogger
            MLflowLogger(active=True).log_artifact(plot_path, artifact_path="plots")

            print("✅ History loaded and logged to MLflow successfully.")
    else:
        print(f"⚠️ No history file found at {history_path}")

class MLflowLogger:
    def __init__(self, active: bool = True):
        self.active = active and (mlflow is not None)

    def log_artifact(self, path: str, artifact_path: str = None):
        if not self.active:
            return
        if artifact_path:
            mlflow.log_artifact(path, artifact_path=artifact_path)
        else:
            mlflow.log_artifact(path)
    
    def log_metric(self, key: str, value, step: int = None):
        if not self.active:
            return
        if step is not None:
            mlflow.log_metric(key, value, step=step)
        else:
            mlflow.log_metric(key, value)

    def log_metrics(self, metrics: dict):
        if not self.active:
            return
        mlflow.log_metrics(metrics)

    def log_param(self, key: str, value):
        if not self.active:
            return
        mlflow.log_param(key, value)

    def log_params(self, params: dict):
        if not self.active:
            return
        mlflow.log_params(params)

    def start_run(self, **kwargs):
        if not self.active:
            return None
        return mlflow.start_run(**kwargs)

    def end_run(self):
        if not self.active:
            return
        mlflow.end_run()
