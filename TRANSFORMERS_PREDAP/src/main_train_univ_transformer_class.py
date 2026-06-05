"""
Main Training and Evaluation Script for Univariate Transformer - Class-Based
============================================================================
This script provides a class-based approach for orchestrating the entire training and evaluation pipeline.
"""

import os
import sys
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any

import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.losses import Huber

# Import data preparation module
import sys
import os

# Add the src directory to path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir) if os.path.basename(current_dir) != 'src' else current_dir
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from utils import data_preparation
from univariate_transformer.evaluation_univ_transformer import evaluate_univ_transformer

from config import BaseTransformerConfig

from univariate_transformer import (
    build_model, CustomCosineDecay,
    train_given_model_and_data, setup_gpu_memory, create_model_directories, create_pandemic_waves_df, load_and_preprocess_data
)


@dataclass
class TransformerUnivConfig(BaseTransformerConfig):
    """Configuration class for transformer training parameters"""
    
    # Configuration object reference
    config_object: Optional[Any] = field(default=None)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "lookback": self.lookback,
            "forecast": self.forecast,
            "code": self.code,
            "head_size": self.head_size,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "num_transformer_blocks": self.num_transformer_blocks,
            "mlp_units": self.mlp_units,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "cutoff_date": self.cutoff_date,
            "covid_token": self.covid_token,
            "positional_encoding": self.positional_encoding,
            "activation_function": self.activation_function,
            "evaluate_model": self.evaluate_model, 
            "data_path": self.data_path,
        }
    
    def print_config(self):
        """Print configuration in a readable format"""
        print("\n" + "="*60)
        print("TRANSFORMER TRAINING CONFIGURATION")
        print("="*60)
        for key, value in self.to_dict().items():
            print(f"{key:25}: {value}")
        print("="*60)
    
    def get_lr_schedule_params(self, learning_rate=None):
        """Get learning rate schedule parameters."""
        if learning_rate is not None:
            lr_init = learning_rate
        else:
            lr_init = self.learning_rate
        lr_max = lr_init * self.lr_max_multiplier
        lr_min = lr_init * self.lr_min_multiplier
        warmup_steps = int(self.epochs * self.lr_warmup_ratio)
        
        return {
            'initial_lr': lr_init,
            'max_lr': lr_max,
            'min_lr': lr_min,
            'warmup_steps': warmup_steps,
            'total_steps': self.epochs
        }

@dataclass
class UnivariateTransformerPipelineOutputs:
    model: tf.keras.Model
    model_name: str
    train_predictions: np.ndarray
    test_predictions: Optional[np.ndarray]
    loss: Optional[float]
    mae: Optional[float]
    mse: Optional[float]
    rmse: Optional[float]
    wape: Optional[float]

class UnivariateTransformerPipeline:
    """
    Main class for managing the complete univariate transformer training and evaluation pipeline.
    """
    
    def __init__(self, config: TransformerUnivConfig):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: TransformerUnivConfig object containing all parameters
        """
        self.config = config
        self.model = None
        self.model_name = None
        self.training_history = None
        self.evaluation_results = None
        self.data_prep_time = 0
        self.diagnostic_covariates_list = None
        self.train_predictions = None
        self.test_predictions = None
        
        # Initialize paths
        self.data_path =self.config.data_path
        self.plots_dir = self.config.plots_dir
        
    def setup_environment(self):
        """Setup GPU memory and create necessary directories"""
        print("Setting up environment...")
        setup_gpu_memory()
        create_model_directories()
        print("Environment setup complete!")

    def load_diagnostic_covariates(self):
        diagnostic_covariates_path = self.config.diagnostic_covariates_path + self.config.code + ".xlsx"
        diagnostic_covariates_df = pd.read_excel(diagnostic_covariates_path, engine='openpyxl')
        self.diagnostic_covariates_list = list(diagnostic_covariates_df[diagnostic_covariates_df['LAG'] == self.config.forecast]['predictors'])[0].split(',')

        return self.diagnostic_covariates_list
    
    def prepare_data(self, train: bool=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data using the configuration parameters.
        
        Returns:
            Tuple of (X, Y) training data arrays
        """
        print("\n" + "="*50)
        print("DATA PREPARATION PHASE")
        print("="*50)
        
        start_time = time.perf_counter()

        self.diagnostic_covariates_list = self.load_diagnostic_covariates()
        
        X, Y = data_preparation.prepare_data(
            self.data_path, 
            self.config.code, 
            self.config.lookback, 
            self.config.forecast,
            covid_token=self.config.covid_token, 
            cutoff_date=self.config.cutoff_date,
            max_date = self.config.final_cutoff_date,
            train=train, 
            debug=True, 
            univariate=True,
            scaler = self.config.scaler,
            eliminate_covid_data = self.config.eliminate_covid_data,
            covid_dates = self.config.covid_dates,
            relevant_feature_cols=self.diagnostic_covariates_list,
            split_ratio=self.config.default_split_ratio
        )
        
        finish_time = time.perf_counter()
        self.data_prep_time = finish_time - start_time
        
        print(f"Data preparation completed!")
        print(f"Time taken: {self.data_prep_time:.2f} seconds")
        print(f"Data shapes - X: {X.shape}, Y: {Y.shape}")
        
        return X, Y
    

    def build_model(self, input_shape: Tuple[int, ...]) -> tf.keras.Model:
        """
        Build the transformer model based on configuration.
        
        Args:
            input_shape: Shape of input data (sequence_length, features)
            
        Returns:
            Compiled Keras model
        """
        print("\n" + "="*50)
        print("MODEL BUILDING PHASE")
        print("="*50)
        
        model = build_model(
            input_shape,
            head_size=self.config.head_size,
            num_heads=self.config.num_heads,
            ff_dim=self.config.ff_dim,
            num_transformer_blocks=self.config.num_transformer_blocks,
            mlp_units=self.config.mlp_units,
            mlp_dropout=self.config.dropout,
            dropout=self.config.dropout,
            n_pred=self.config.forecast,
            activation_function=self.config.activation_function,
            pos_encoding=self.config.positional_encoding,
        )
        
        print("Model architecture:")
        model.summary()
        
        return model
    
    def setup_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """
        Setup training callbacks including learning rate scheduler and early stopping.
        
        Returns:
            List of Keras callbacks
        """
        callbacks = []
        
        # Learning rate scheduler
        lr_params = self.config.get_lr_schedule_params(learning_rate=self.config.learning_rate)
        scheduler = CustomCosineDecay(**lr_params)
        
        lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
        callbacks.append(lr_callback)
        
        # Early stopping
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=self.config.early_stop_patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
        
        return callbacks
    
    def compile_model(self, model: tf.keras.Model) -> tf.keras.Model:
        """
        Compile the model with appropriate optimizer and loss.
        
        Args:
            model: The model to compile
            
        Returns:
            Compiled model
        """
        model.compile(
            loss='mae', 
            metrics=['mae', 'mse'], 
            optimizer=Adam(
                clipnorm = 2.0,
                learning_rate=self.config.learning_rate,
                #use_ema=True
                
            ),
        )
        return model
    
    def train_model(self, X: np.ndarray, Y: np.ndarray) -> Dict[str, Any]:
        """
        Train the transformer model.
        
        Args:
            X: Input training data
            Y: Target training data
            
        Returns:
            Dictionary containing training results
        """
        print("\n" + "="*50)
        print("TRAINING PHASE")
        print("="*50)
        
        # Build model if not already built
        if self.model is None:
            self.model = self.build_model((self.config.lookback, X.shape[-1]))
            self.model = self.compile_model(self.model)
        
        # Generate model name
        self.model_name = self.config.get_model_name()
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Train the model
        training_results = train_given_model_and_data(
            self.model, X, Y,
            batch_size=self.config.batch_size,
            model_name=self.model_name,
            epochs=self.config.epochs,
            save_model=True,
            save_memory=False,
            callbacks=callbacks,
            save_history=self.config.save_train_history,
            shuffle=self.config.shuffle_data,
            patience = self.config.early_stop_patience,
        )
        
        self.training_history = training_results

        self.train_predictions = self.model.predict(X, verbose=1)
        
        print("Training completed successfully!")
        return training_results
    
    def evaluate_model(self) -> Optional[Tuple[float, float, float]]:
        """
        Evaluate the trained model if evaluation is enabled.
        
        Returns:
            Tuple of (loss, mae, mse) if evaluation is performed, None otherwise
        """
        if not self.config.evaluate_model or self.model_name is None:
            return None
            
        print("\n" + "="*50)
        print("EVALUATION PHASE")
        print("="*50)
        
        print("Files in model folder:", os.listdir(self.config.model_folder))
        
        trained_models = [f for f in os.listdir(self.config.model_folder) if f.endswith('.keras')]
        print("Trained models:", trained_models)
        
        # Create pandemic waves DataFrame
        df_waves = create_pandemic_waves_df()
        
        # Evaluate the model
        predictions, loss, mae, mse, rmse, wape = evaluate_univ_transformer(
            self.model_name,
            self.config.data_path,
            self.config.code,
            cutoff_date=self.config.cutoff_date,
            max_date = self.config.final_cutoff_date,
            covid_token=self.config.covid_token,
            MODEL_FOLDER=self.config.model_folder,
            df_waves=df_waves,
            scaler = self.config.scaler,
            eliminate_covid_data = self.config.eliminate_covid_data,
            covid_dates = self.config.covid_dates,
            relevant_feature_cols=self.diagnostic_covariates_list,
            batch_size=self.config.batch_size,
            split_ratio=self.config.default_split_ratio,
            
        )
        
        self.evaluation_results = {
            "loss": loss,
            "mae": mae, 
            "mse": mse,
            "rmse": rmse,
            "wape": wape
        }
        
        print(f"Evaluation results:")
        print(f"  Loss: {loss:.6f}")
        print(f"  MAE:  {mae:.6f}")
        print(f"  MSE:  {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  WAPE: {wape:.6f}%")
        
        return predictions, loss, mae, mse, rmse, wape
    
    def run_complete_pipeline(self) -> Tuple[tf.keras.Model, str, Optional[float], Optional[float], Optional[float]]:
        """
        Run the complete training and evaluation pipeline.
        
        Returns:
            Tuple of (model, model_name, loss, mae, mse)
        """
        # Print configuration
        self.config.print_config()
        
        # Setup environment
        self.setup_environment()
        
        # Prepare data
        X, Y = self.prepare_data()
        
        # Train model
        self.train_model(X, Y)
        
        # Evaluate model
        evaluation_results = self.evaluate_model()
        
        # Extract evaluation metrics
        test_predictions, loss, mae, mse, rmse, wape = None, None, None, None, None, None
        if evaluation_results is not None:
            self.test_predictions, loss, mae, mse, rmse, wape = evaluation_results
        
        print("\n" + "="*50)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Model name: {self.model_name}")
        print(f"Data preparation time: {self.data_prep_time:.2f} seconds")
        if evaluation_results:
            print(f"Final metrics - Loss: {loss:.6f}, MAE: {mae:.6f}, MSE: {mse:.6f}, RMSE: {rmse:.6f}, WAPE: {wape:.6f}%")
        
        #return self.model, self.model_name, self.train_predictions, self.test_predictions, loss, mae, mse, rmse, wape
        return UnivariateTransformerPipelineOutputs(
            model=self.model,
            model_name=self.model_name,
            train_predictions=self.train_predictions,
            test_predictions=self.test_predictions,
            loss=loss,
            mae=mae,
            mse=mse,
            rmse=rmse,
            wape=wape
        )
    
    def get_results_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of pipeline results.
        
        Returns:
            Dictionary containing all results and configuration
        """
        return {
            "config": self.config.to_dict(),
            "model_name": self.model_name,
            "training_history": self.training_history,
            "evaluation_results": self.evaluation_results,
            "data_preparation_time": self.data_prep_time,
            "model_summary": self.model.summary() if self.model else None
        }


# Convenience function for backward compatibility
def main_univ_transformer(lookback: int, forecast: int, code: str, **kwargs) -> Tuple[tf.keras.Model, str, Optional[float], Optional[float], Optional[float]]:
    """
    Main function that orchestrates the training and evaluation pipeline using the class-based approach.
    This function maintains backward compatibility while using the new OOP structure internally.
    
    Args:
        lookback: Number of past time steps to use for prediction
        forecast: Number of future time steps to predict
        code: Target code for data filtering
        **kwargs: Additional configuration parameters
        
    Returns:
        Tuple of (model, model_name, loss, mae, mse)
    """
    
    # Create configuration object
    config = TransformerUnivConfig(
        lookback=lookback,
        forecast=forecast,
        code=code,
        **kwargs
    )
    
    # Create and run pipeline
    pipeline = UnivariateTransformerPipeline(config)
    return pipeline.run_complete_pipeline()


def run_batch_experiments(configs: List[TransformerUnivConfig]) -> List[Dict[str, Any]]:
    """
    Run multiple experiments with different configurations.
    
    Args:
        configs: List of TransformerTrainingConfig objects
        
    Returns:
        List of result summaries
    """
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"RUNNING EXPERIMENT {i+1}/{len(configs)}")
        print(f"{'='*60}")
        
        pipeline = UnivariateTransformerPipeline(config)
        pipeline.run_complete_pipeline()
        
        results.append(pipeline.get_results_summary())
    
    return results


if __name__ == "__main__":
    # Example 1: Using the class-based approach directly
    transformer_config = TransformerUnivConfig(
                            lookback=7,
                            forecast=7,
                            code="T14",
                            activation_function='gelu',
                            cutoff_date="2008-01-01",
                            evaluate_model = True,
                            positional_encoding = False,
                            
                            )

    pipeline = UnivariateTransformerPipeline(transformer_config)
    model, model_name, loss, mae, mse, rmse, wape = pipeline.run_complete_pipeline()
    