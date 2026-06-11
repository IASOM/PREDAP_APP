"""
Main Training and Evaluation Script for Seasonal Residual Transformer - Class-Based
===================================================================================
This script provides a class-based approach for training seasonal residual transformers
that learn to correct base model predictions using seasonal covariates.
"""

import os
import sys
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any

from tensorflow.keras.losses import Huber
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from univariate_transformer import CustomCosineDecay
from utils.experiments_utils import smart_read

# Add the src directory to path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir) if os.path.basename(current_dir) != 'src' else current_dir
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import necessary modules
from utils import data_preparation
from config.base_transformer_config import BaseTransformerConfig

from src.residual_multivariate_transformers.model_architecture_residual_transformer import (
    RevIN,
    PositionalEncoding,
    CustomCosineDecay,
)

from residual_multivariate_transformers import (
    # Model architecture and training
    hybrid_lstm_transformer_model, train_given_model_and_data,
    setup_gpu_memory, create_model_directories, load_trained_model,
    
    # Data preparation utilities
    prepare_base_model_data, load_base_model_transformer, prepare_residual_data,
    split_train_test, create_pandemic_waves_df,
    
    # Visualization and evaluation
    plot_stepwise_errors_comparison, plot_residuals_analysis,
    plot_predictions_with_pandemic_waves, plot_errors_over_time_with_waves,
    evaluate_error_significance_pandemic_waves, save_performance_results,
    compare_model_performance
)




@dataclass
class SeasonalResidualTransformerConfig(BaseTransformerConfig):
    """Configuration class for seasonal residual transformer training parameters"""
    
    # Configuration object reference  
    config_object: Optional[Any] = field(default=None)
    predictions_train_corrected: Optional[Any] = None
    predictions_test_corrected: Optional[Any] = None
    categorical_vars: List[str] = field(default_factory=lambda: ["Day_of_Week", "Month", "Season", "Holiday", "School_Vacation", "Is_Weekend"])

    # Visualization and evaluation flags
    plot_stepwise_errors: bool = field(default=True)
    plot_residuals_analysis: bool = field(default=True)
    plot_pandemic_waves: bool = field(default=True)
    plot_errors_over_time: bool = field(default=True)
    evaluate_error_significance: bool = field(default=True)
    save_performance_results: bool = field(default=True)

    
    def print_config(self):
        """Print configuration in a readable format"""
        print("\n" + "="*60)
        print("SEASONAL RESIDUAL TRANSFORMER CONFIGURATION")
        print("="*60)
        for key, value in self.to_dict().items():
            if isinstance(value, list) and len(value) > 5:
                print(f"{key:35}: [{', '.join(map(str, value[:3]))}, ... ({len(value)} total)]")
            else:
                print(f"{key:35}: {value}")
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
class SeasonalResidualPipelineOutputs:
    predictions_train_corrected: np.ndarray
    predictions_test_corrected: np.ndarray
    residual_diagnostics_model: Any
    residual_diagnostics_model_name: str
    corrected_diagnostics_mae: float
    corrected_diagnostics_mse: float
    corrected_diagnostics_rmse: float
    corrected_diagnostics_wape: float

class SeasonalResidualTransformerPipeline:
    """
    Main class for managing the complete seasonal residual transformer training and evaluation pipeline.
    This pipeline trains a transformer to learn residual corrections using seasonal covariates.
    """
    
    def __init__(self, config: SeasonalResidualTransformerConfig):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: SeasonalResidualTransformerConfig object containing all parameters
        """
        self.config = config
        self.residual_model = None
        self.base_model_name = None
        self.residual_model_name = None
        
        # Data containers
        self.Y_train = None
        self.Y_test = None
        self.X_train = None
        self.X_test = None
        self.date_list_train = None
        self.date_list_test = None
        
        # Residual data containers
        self.Y_train_residual = None
        self.Y_test_residual = None
        self.X_train_covs = None
        self.X_test_covs = None
        
        # Predictions containers
        self.predictions_train = None
        self.predictions_test = None
        self.predictions_train_corrected = None
        self.predictions_test_corrected = None
        
        # Results containers
        self.evaluation_metrics = None
        self.training_history = None
        
        # Initialize paths
        self.data_path = self.config.data_path
        
    def setup_environment(self):
        """Setup GPU memory and create necessary directories"""
        print("="*60)
        print("SEASONAL RESIDUAL MULTIVARIATE TRANSFORMER PIPELINE")
        print("="*60)
        print(f"Target Code: {self.config.code}")
        print(f"Forecast Horizon: {self.config.forecast}")
        print(f"Lookback Window: {self.config.lookback}")
        
        setup_gpu_memory()
        create_model_directories()
        
    def prepare_base_model_data(self):
        """Prepare data for base model and load base model predictions"""
        print("\n" + "="*50)
        print("PHASE 1: LOADING BASE MODEL AND COMPUTING RESIDUALS")
        print("="*50)
        
        # Generate model names
        self.base_model_name = self.config.get_model_name()
        self.residual_model_name = self.config.get_seasonal_residual_model_name()
        
        print(f"Base Model: {self.base_model_name}")
        print(f"Residual Model: {self.residual_model_name}")
        
        # Load and prepare data for the base model
        print("Preparing data for base model...")
        
        start_time = time.perf_counter()
        
        self.Y_train, self.Y_test, self.X_train, self.X_test, self.date_list_train, self.date_list_test = prepare_base_model_data(
            self.data_path, 
            self.config.code, 
            self.config.lookback, 
            self.config.forecast, 
            covid_token=self.config.covid_token, 
            cutoff_date=self.config.cutoff_date,
            max_date = self.config.final_cutoff_date,
            scaler = self.config.scaler,
            eliminate_covid_data=self.config.eliminate_covid_data,
            covid_dates=self.config.covid_dates,
            split_ratio=self.config.default_split_ratio

        )
        
        # Load base model predictions unless a complete corrected train/test pair was provided.
        if (
            self.config.predictions_train_corrected is None
            or self.config.predictions_test_corrected is None
        ):
            self.predictions_train, self.predictions_test = load_base_model_transformer(
                self.X_train, 
                self.X_test, 
                self.config.model_folder, 
                self.base_model_name
            )
        else:
            self.predictions_train = self.config.predictions_train_corrected
            self.predictions_test = self.config.predictions_test_corrected
        
        print(f"Base model predictions - Test: {self.predictions_test.shape}, Train: {self.predictions_train.shape}")
        print(f"Actual values - Test: {self.Y_test.shape}, Train: {self.Y_train.shape}")
        
        # Compute residuals for train and test sets
        print("\nComputing residuals...")
        self.Y_train_residual = prepare_residual_data(self.predictions_train, self.Y_train)
        self.Y_test_residual = prepare_residual_data(self.predictions_test, self.Y_test)
        
        print(f"Residuals shape - Train: {self.Y_train_residual.shape}, Test: {self.Y_test_residual.shape}")
        
    def prepare_covariate_data(self):
        """Prepare seasonal covariate data for residual model"""
        print("\n" + "="*50)
        print("PHASE 2: PREPARING COVARIATE DATA FOR RESIDUAL MODEL")
        print("="*50)
        
        
        df = smart_read(self.data_path)
        # Prepare seasonal features for training data
        print("Preparing seasonal features for training data...")
        df_processed = data_preparation.prepare_time_series_features(
            df, 
            self.config.categorical_vars, 
            cutoff_date=self.config.cutoff_date,
            max_date = self.config.final_cutoff_date,
            scaler = self.config.scaler,
            eliminate_covid_data=self.config.eliminate_covid_data, 
            covid_dates=self.config.covid_dates,
        )

        # Load and split the original data for covariate extraction
        df_train_processed, df_test_processed = split_train_test(
            df_processed, 
            split_ratio=self.config.default_split_ratio, 
            cutoff_date=self.config.cutoff_date,
            max_date = self.config.final_cutoff_date,

        )
        
        # Generate rolling sequences with covariates for training
        print("Generating rolling sequences with seasonal covariates for training...")
        self.X_train_covs = data_preparation.generate_rolling_sequences_covariates(
            df_train_processed, 
            self.config.lookback, 
            self.config.forecast, 
            self.predictions_train, 
        )
        
        print(f"Training covariates shape: {self.X_train_covs.shape}")
        print(f"Expected shape: (num_samples, {self.config.lookback}, num_features)")
        
        # Generate rolling sequences for test data
        print("Generating rolling sequences with seasonal covariates for test data...")
        self.X_test_covs = data_preparation.generate_rolling_sequences_covariates(
            df_test_processed, 
            self.config.lookback, 
            self.config.forecast, 
            self.predictions_test
        )
        
        print(f"Test covariates shape: {self.X_test_covs.shape}")

        return self.X_train_covs, self.X_test_covs
        
    def build_residual_model(self) -> tf.keras.Model:
        """Build the hybrid LSTM + Transformer model for seasonal residuals"""
        print("\n" + "="*50)
        print("PHASE 3: TRAINING RESIDUAL CORRECTION MODEL")
        print("="*50)
        
        print("Building residual correction model...")
        
        transformer_params = {
            'head_size': self.config.head_size,
            'num_heads': self.config.num_heads,
            'ff_dim': self.config.ff_dim,
            'mlp_units': self.config.mlp_units,
            'num_transformer_blocks': self.config.num_transformer_blocks,
            'dropout': self.config.dropout,
        }
        
        # Build the residual model
        self.residual_model = hybrid_lstm_transformer_model(
            input_shape=(self.config.forecast, self.X_train_covs.shape[2]), 
            forecast=self.config.forecast,
            activation_function=self.config.activation_function,
            transformer_params=transformer_params
        )

        self.residual_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.learning_rate,
                clipnorm= 1.0,
                #use_ema=True
                ), 
            loss='mae', 
            metrics=['mae', 'mse']
        )
        
        print("Residual model architecture:")
        self.residual_model.summary()
        
        return self.residual_model

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


    def train_residual_model(self):
        """Train the residual correction model"""
        if self.residual_model is None:
            self.build_residual_model()
        
        # Create structured directory: model_folder/{code}/seasonal_model/
        model_save_folder = os.path.join(self.config.model_folder, self.config.code, "seasonal_model")
        os.makedirs(model_save_folder, exist_ok=True)
            
        print("Training residual correction model...")
        callbacks = self.setup_callbacks()
        # Train the residual model
        self.training_history = train_given_model_and_data(
            self.residual_model, 
            self.X_train_covs, 
            self.Y_train_residual,
            batch_size=self.config.batch_size,
            model_name=self.residual_model_name,
            epochs=self.config.epochs,
            save_model=True,
            save_memory=False,
            callbacks=callbacks,
            save_history=True,
            model_folder=model_save_folder,
        )
        metadata_path = self.config.save_metadata(self.residual_model_name, model_folder=model_save_folder)
        print(f"Residual model metadata saved to: {metadata_path}")
        
        # Generate corrected training predictions
        predicted_residuals_train = self.residual_model.predict(self.X_train_covs, verbose=1)
        #predicted_residuals_train = np.squeeze(predicted_residuals_train, axis=-1)
        self.predictions_train_corrected = self.predictions_train + predicted_residuals_train
        
    def evaluate_residual_model(self):
        """Evaluate the residual correction model on test data"""
        print("\n" + "="*50)
        print("PHASE 4: EVALUATING RESIDUAL CORRECTION MODEL")
        print("="*50)
        
        # Load the trained residual model if it was saved and reloaded
        model_path = os.path.join(self.data_path, self.residual_model_name)
        if os.path.exists(model_path):
            self.residual_model = load_trained_model(model_path)
        
        # Predict residuals for the test set
        print("Predicting residuals for test set...")
        predicted_residuals = self.residual_model.predict(self.X_test_covs, verbose=1)
        predicted_residuals = np.squeeze(predicted_residuals, axis=-1) if predicted_residuals.shape[-1] == 1 else predicted_residuals
        
        # Correct the original forecast
        print("Computing corrected forecasts...")
        self.predictions_test_corrected = self.predictions_test + predicted_residuals
        
        return self.predictions_test_corrected
        
    def generate_visualizations(self):
        """Generate comprehensive visualizations and analysis"""
        print("\n" + "="*50)
        print("PHASE 5: VISUALIZATION AND ANALYSIS")
        print("="*50)

        # Get original scale data for visualization
        original_scale_df = smart_read(self.data_path)

        test_timestamp = data_preparation.extract_dates(self.data_path, 
                                       self.config.code, 
                                       self.config.lookback, 
                                       self.config.forecast, 
                                       train=False, 
                                       cutoff_date=self.config.cutoff_date, 
                                       max_date=self.config.final_cutoff_date,
                                       eliminate_covid_data=self.config.eliminate_covid_data,
                                       covid_dates=self.config.covid_dates
                                       )
        
        # Get original scale test data
        X_test_orig, Y_test_orig = data_preparation.prepare_data_not_normalized(
            self.data_path, 
            self.config.code, 
            self.config.lookback, 
            self.config.forecast,
            covid_token=self.config.covid_token, 
            cutoff_date=self.config.cutoff_date, 
            max_date=self.config.final_cutoff_date,
            train=False, 
            univariate=True,
            eliminate_covid_data=self.config.eliminate_covid_data, 
            covid_dates=self.config.covid_dates,
            split_ratio = self.config.default_split_ratio
        )
        
        # Inverse transform predictions
        corrected_forecast_orig = data_preparation.inverse_transform_predictions(
            self.predictions_test_corrected, 
            original_scale_df, 
            self.config.code, 
            forecast=self.config.forecast,
            lookback=self.config.lookback, 
            cutoff_date=self.config.cutoff_date,
            max_date=self.config.final_cutoff_date,
            scaler = self.config.scaler,
            eliminate_covid_data=self.config.eliminate_covid_data, 
            covid_dates=self.config.covid_dates

        )
        corrected_forecast_orig = np.maximum(corrected_forecast_orig, 0)  # Ensure no negative predictions
        
        predictions_test_orig = data_preparation.inverse_transform_predictions(
            self.predictions_test, 
            original_scale_df, 
            self.config.code, 
            forecast=self.config.forecast,
            lookback=self.config.lookback, 
            cutoff_date=self.config.cutoff_date,
            max_date=self.config.final_cutoff_date,
            scaler = self.config.scaler,
            eliminate_covid_data=self.config.eliminate_covid_data, 
            covid_dates=self.config.covid_dates
        )
        predictions_test_orig = np.maximum(predictions_test_orig, 0)  # Ensure no negative predictions
        
        print(f"Corrected forecast shape: {corrected_forecast_orig.shape}")
        
        if self.config.plot_stepwise_errors:
            print("Plotting stepwise errors comparison...")
            plot_stepwise_errors_comparison(
                self.Y_test, 
                self.predictions_test, 
                corrected_forecast_orig, 
                f"{self.config.code} Seasonal Residual Correction", 
                model_name=self.residual_model_name
            )
        
        if self.config.plot_residuals_analysis:
            print("Plotting residuals analysis...")
            plot_residuals_analysis(
                predictions_test_orig, 
                corrected_forecast_orig, 
                Y_test_orig, 
                f"{self.config.code} Seasonal Residual Correction", 
                model_name=self.residual_model_name,
                timestamp = test_timestamp
            )
        
        # Create pandemic waves DataFrame
        df_waves = create_pandemic_waves_df()
        
        # Prepare data for plotting (average across forecast horizon if needed)
        predictions_to_plot = predictions_test_orig.mean(axis=1) if len(predictions_test_orig.shape) > 2 else predictions_test_orig
        corrected_to_plot = corrected_forecast_orig.mean(axis=1) if len(corrected_forecast_orig.shape) > 2 else corrected_forecast_orig
        Y_test_to_plot = Y_test_orig.mean(axis=1) if len(Y_test_orig.shape) > 2 else Y_test_orig
        
        if self.config.plot_pandemic_waves:
            print("Plotting predictions with pandemic waves...")
            plot_predictions_with_pandemic_waves(
                Y_test_to_plot, 
                predictions_to_plot, 
                self.date_list_test, 
                df_waves, 
                model_name=self.residual_model_name
            )
            
            plot_predictions_with_pandemic_waves(
                Y_test_to_plot, 
                corrected_to_plot, 
                self.date_list_test, 
                df_waves, 
                model_name=self.residual_model_name
            )
        
        if self.config.plot_errors_over_time:
            print("Plotting errors over time with pandemic waves...")
            plot_errors_over_time_with_waves(
                Y_test_to_plot, 
                predictions_to_plot, 
                self.date_list_test, 
                df_waves
            )
            
            plot_errors_over_time_with_waves(
                Y_test_to_plot, 
                corrected_to_plot, 
                self.date_list_test, 
                df_waves
            )
        
        if self.config.evaluate_error_significance:
            print("Evaluating error significance during pandemic waves...")
            print("Original predictions:")
            evaluate_error_significance_pandemic_waves(
                Y_test_to_plot, 
                predictions_to_plot, 
                self.date_list_test, 
                df_waves
            )
            
            print("Corrected predictions:")
            evaluate_error_significance_pandemic_waves(
                Y_test_to_plot, 
                corrected_to_plot, 
                self.date_list_test, 
                df_waves
            )
        
        return predictions_to_plot, corrected_to_plot, Y_test_to_plot
        
    def calculate_performance_metrics(self, predictions_to_plot, corrected_to_plot, Y_test_to_plot):
        """Calculate and display performance metrics"""
        print("\n" + "="*50)
        print("PHASE 6: PERFORMANCE SUMMARY")
        print("="*50)
        
        # Original model metrics
        original_mae = mean_absolute_error(Y_test_to_plot, predictions_to_plot)
        original_mse = mean_squared_error(Y_test_to_plot, predictions_to_plot)
        original_rmse = np.sqrt(original_mse)
        original_wape = np.sum(np.abs(Y_test_to_plot - predictions_to_plot)) / np.sum(np.abs(Y_test_to_plot)) * 100
        
        # Corrected model metrics
        corrected_mae = mean_absolute_error(Y_test_to_plot, corrected_to_plot)
        corrected_mse = mean_squared_error(Y_test_to_plot, corrected_to_plot)
        corrected_rmse = np.sqrt(corrected_mse)
        corrected_wape = np.sum(np.abs(Y_test_to_plot - corrected_to_plot)) / np.sum(np.abs(Y_test_to_plot)) * 100
        
        # Store metrics
        self.evaluation_metrics = {
            "original_mae": original_mae,
            "original_mse": original_mse,
            "original_rmse": original_rmse,
            "original_wape": original_wape,
            "corrected_mae": corrected_mae,
            "corrected_mse": corrected_mse,
            "corrected_rmse": corrected_rmse,
            "corrected_wape": corrected_wape,
            "mae_improvement": ((original_mae - corrected_mae) / original_mae * 100),
            "mse_improvement": ((original_mse - corrected_mse) / original_mse * 100),
            "rmse_improvement": ((original_rmse - corrected_rmse) / original_rmse * 100),
            "wape_improvement": ((original_wape - corrected_wape) / original_wape * 100),
        }
        
        print("PERFORMANCE COMPARISON:")
        print("-" * 40)
        print(f"Original Model:")
        print(f"  MAE:  {original_mae:.6f}")
        print(f"  MSE:  {original_mse:.6f}")
        print(f"  RMSE: {original_rmse:.6f}")
        print(f"  WAPE: {original_wape:.6f}")
        print()
        print(f"Seasonal Residual Corrected Model:")
        print(f"  MAE:  {corrected_mae:.6f}")
        print(f"  MSE:  {corrected_mse:.6f}")
        print(f"  RMSE: {corrected_rmse:.6f}")
        print(f"  WAPE: {corrected_wape:.6f}")
        print()
        print(f"IMPROVEMENT:")
        print(f"  MAE:  {self.evaluation_metrics['mae_improvement']:+.2f}%")
        print(f"  MSE:  {self.evaluation_metrics['mse_improvement']:+.2f}%")
        print(f"  RMSE: {self.evaluation_metrics['rmse_improvement']:+.2f}%")
        print(f"  WAPE: {self.evaluation_metrics['wape_improvement']:+.2f}%")
        
        # Save performance results to JSON
        if self.config.save_performance_results:
            save_performance_results(
                model_name=self.residual_model_name,
                original_mae=original_mae,
                original_mse=original_mse, 
                original_rmse=original_rmse,
                original_wape=original_wape,
                corrected_mae=corrected_mae,
                corrected_mse=corrected_mse,
                corrected_rmse=corrected_rmse,
                corrected_wape=corrected_wape,
                forecast=self.config.forecast,
                lookback=self.config.lookback,
                code=self.config.code
            )
        
        # Compare with other models
        try:
            compare_model_performance()
        except Exception as e:
            print(f"Note: Could not compare model performance: {e}")
        
        return corrected_mae, corrected_mse, corrected_rmse, corrected_wape
        
    def run_complete_pipeline(self) -> Tuple[np.ndarray, np.ndarray, tf.keras.Model, str, float, float, float]:
        """
        Run the complete seasonal residual transformer training and evaluation pipeline.
        
        Returns:
            Tuple of (predictions_train_corrected, predictions_test_corrected, 
                     residual_model, residual_model_name, corrected_mae, corrected_mse, corrected_rmse)
        """
        # Print configuration
        self.config.print_config()
        
        # Setup environment
        self.setup_environment()
        
        # Phase 1: Prepare base model data and compute residuals
        self.prepare_base_model_data()
        
        # Phase 2: Prepare seasonal covariate data for residual model
        X_train_covs, X_test_covs = self.prepare_covariate_data()
        
        # Phase 3: Build and train residual model
        self.train_residual_model()
        
        # Phase 4: Evaluate residual model
        self.evaluate_residual_model()
        
        # Phase 5: Generate visualizations
        predictions_to_plot, corrected_to_plot, Y_test_to_plot = self.generate_visualizations()
        
        predictions_to_plot = np.maximum(predictions_to_plot, 0)  # Ensure no negative predictions
        corrected_to_plot = np.maximum(corrected_to_plot, 0)  # Ensure no negative predictions
        Y_test_to_plot = np.maximum(Y_test_to_plot, 0)  # Ensure no negative predictions
        # Phase 6: Calculate performance metrics
        corrected_mae, corrected_mse, corrected_rmse, corrected_wape = self.calculate_performance_metrics(
            predictions_to_plot, corrected_to_plot, Y_test_to_plot
        )
        
        print("\n" + "="*50)
        print("SEASONAL RESIDUAL MULTIVARIATE TRANSFORMER PIPELINE COMPLETE")
        print("="*50)
        
        '''return (self.predictions_train_corrected, self.predictions_test_corrected, 
                self.residual_model, self.residual_model_name, 
                corrected_mae, corrected_mse, corrected_rmse, corrected_wape)'''
        
        return SeasonalResidualPipelineOutputs(
            predictions_train_corrected=self.predictions_train_corrected,
            predictions_test_corrected=self.predictions_test_corrected,
            residual_diagnostics_model=self.residual_model,
            residual_diagnostics_model_name=self.residual_model_name,
            corrected_diagnostics_mae=corrected_mae,
            corrected_diagnostics_mse=corrected_mse,
            corrected_diagnostics_rmse=corrected_rmse,
            corrected_diagnostics_wape=corrected_wape
        )
    
    def get_results_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of pipeline results.
        
        Returns:
            Dictionary containing all results and configuration
        """
        return {
            "config": self.config.to_dict(),
            "residual_model_name": self.residual_model_name,
            "base_model_name": self.base_model_name,
            "evaluation_metrics": self.evaluation_metrics,
            "categorical_vars_used": self.config.categorical_vars,
            "data_shapes": {
                "Y_train": self.Y_train.shape if self.Y_train is not None else None,
                "Y_test": self.Y_test.shape if self.Y_test is not None else None,
                "X_train_covs": self.X_train_covs.shape if self.X_train_covs is not None else None,
                "X_test_covs": self.X_test_covs.shape if self.X_test_covs is not None else None,
            }
        }


def main():
    """Main function to run seasonal residual transformer with default configuration"""
    # Create default configuration

    
    config = SeasonalResidualTransformerConfig()
    
    # Create and run pipeline
    pipeline = SeasonalResidualTransformerPipeline(config)
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline()
    
    predictions_train_corrected, predictions_test_corrected, residual_model, residual_model_name, corrected_mae, corrected_mse, corrected_rmse = results
    
    # Print summary
    print(f"\nPipeline completed successfully!")
    print(f"Residual Model: {residual_model_name}")
    print(f"Final Metrics - MAE: {corrected_mae:.6f}, MSE: {corrected_mse:.6f}, RMSE: {corrected_rmse:.6f}")
    
    return results


if __name__ == "__main__":
    main()
