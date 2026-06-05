import os
import pandas as pd
from typing import List
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from model_architechture.layers import CustomCosineDecay
from training.training_residual_transformer import load_trained_model

def load_diagnostic_covariates(config, code, forecast):
    """
    Load diagnostic covariates from Excel extracted from the CCLR pipeline.
    Args:        config: Configuration object containing paths and parameters
    Returns:     List of diagnostic covariate names
    """
    diagnostic_covariates_path = config.diagnostic_covariates_path + code + ".xlsx"
    diagnostic_covariates_df = pd.read_excel(diagnostic_covariates_path, engine='openpyxl')
    diagnostic_covariates_list = list(diagnostic_covariates_df[diagnostic_covariates_df['LAG'] == forecast]['predictors'])[0].split(',')

    return diagnostic_covariates_list


def get_lr_schedule_params(config, learning_rate=None):
        """Get learning rate schedule parameters."""
        if learning_rate is not None:
            lr_init = learning_rate
        else:
            lr_init = config.learning_rate
        lr_max = lr_init * config.lr_max_multiplier
        lr_min = lr_init * config.lr_min_multiplier
        warmup_steps = int(config.epochs * config.lr_warmup_ratio)
        
        return {
            'initial_lr': lr_init,
            'max_lr': lr_max,
            'min_lr': lr_min,
            'warmup_steps': warmup_steps,
            'total_steps': config.epochs
        }

def setup_callbacks(config, learning_rate = None, early_stop_patience = None) -> List[tf.keras.callbacks.Callback]:
    """
    Setup training callbacks including learning rate scheduler and early stopping.
    
    Returns:
        List of Keras callbacks
    """
    callbacks = []

    if learning_rate is None:
        learning_rate = config.learning_rate
    if early_stop_patience is None:
        early_stop_patience = config.early_stop_patience
    
    # Learning rate scheduler
    lr_params = get_lr_schedule_params(config,learning_rate=learning_rate)
    scheduler = CustomCosineDecay(**lr_params)
    
    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    callbacks.append(lr_callback)
    
    # Early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=early_stop_patience,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stop)
    
    return callbacks


def compile_model(config, model: tf.keras.Model, learning_rate: float = None) -> tf.keras.Model:
    """
    Compile the model with appropriate optimizer and loss.
    
    Args:
        config: Configuration object containing training parameters
        model: The model to compile
        learning_rate: The learning rate for the optimizer

    Returns:
        Compiled model
    """

    if learning_rate is None:
        learning_rate = config.learning_rate

    model.compile(
        loss='mae', 
        metrics=['mae', 'mse'], 
        optimizer=Adam(
            clipnorm = 2.0,
            learning_rate=learning_rate,
            #use_ema=True
            
        ),
    )
    return model


def load_base_model_transformer(X_train, X_test,base_path, base_model_name):
    """
    Load a trained base transformer model from the specified path.
    
    Parameters:
    -----------
    base_path : str
        Directory path where the model is stored
    base_model_name : str
        Filename of the model to load
    X_train : np.ndarray
        Training input data for prediction
    X_test : np.ndarray
        Testing input data for prediction
        
    Returns:
    --------
    predictions_train : np.ndarray
        Predictions on the training data
    predictions_test : np.ndarray
        Predictions on the testing data
        
    Raises:
    -------
    FileNotFoundError
        If the model file does not exist
    """
    #Load the base transformer model
    available_models = os.listdir(base_path) if os.path.exists(base_path) else []
    
    if base_model_name in available_models:
        full_path = os.path.join(base_path, base_model_name)
        print(f"✅ Found base model: {base_model_name}")
    else:
        print(f"❌ Base model not found: {base_model_name}")
        print(f"Available models: {available_models}")
        return
    
    # Load the base model
    model = load_trained_model(full_path)
    
    # Get predictions from the base transformer model
    print("\nGenerating predictions from base model...")
    
    predictions_train = model.predict(X_train)
    predictions_test = model.predict(X_test)
    
    return predictions_train, predictions_test
