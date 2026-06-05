# training_dl.py
# ===============
# Model training utilities for deep learning models
# Author: Guillem Hernández Guillamet
# Version: 1.0
import tensorflow as tf
from .prediction_dl import prediction, inverse_transform
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def fit_model(model, X_train, y_train, epochs, batch_size, validation_data=None, patience=10, verbose=1):
    """
    Train a deep learning model with callbacks for optimization.
    
    Parameters:
    -----------
    model : tensorflow.keras.Model
        The compiled model to train
    X_train : np.ndarray
        Training input data
    y_train : np.ndarray
        Training target data
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    validation_data : tuple, optional
        Validation data (X_val, y_val)
    patience : int
        Early stopping patience
    verbose : int
        Verbosity level for training output
        
    Returns:
    --------
    tensorflow.keras.callbacks.History
        Training history object
        
    Example:
    --------
    >>> history = fit_model(model, X_train, y_train, epochs=100, batch_size=32)
    """
    try:
        # Define callbacks
        checkpoint_callback = ModelCheckpoint(
            filepath='best_model.keras',
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            save_best_only=True
        )

        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.005,
            patience=patience,
            mode='min'
        )

        rlrop_callback = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.001,
            mode='min'
        )

        # Train the model
        history = model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint_callback, early_stopping_callback, rlrop_callback],
            #verbose=verbose
        )
        
        return history
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        return None
    
def auto_grid_search(model_fn, X_train, y_train, X_test, y_test, scaler, validation_data, patience=10):
    """
    Perform automated grid search for hyperparameter optimization.
    
    Parameters:
    -----------
    model_fn : callable
        Function that creates the model
    X_train : np.ndarray
        Training input data
    y_train : np.ndarray
        Training target data
    X_test : np.ndarray
        Test input data
    y_test : np.ndarray
        Test target data
    scaler : sklearn.preprocessing.MinMaxScaler
        Fitted scaler for inverse transformation
    validation_data : tuple
        Validation data (X_val, y_val)
    patience : int
        Early stopping patience
        
    Returns:
    --------
    dict
        Best configuration found during grid search
        
    Example:
    --------
    >>> best_config = auto_grid_search(create_model_lstm, X_train, y_train, 
    ...                               X_test, y_test, scaler, val_data)
    """
    try:
        # Define hyperparameter grids
        epochs_grid = [10, 20, 50, 100]
        batch_sizes = [16, 32, 64]
        optimizers = ['SGD', 'Adam', 'RMSprop']

        best_mape = float('inf')
        best_config = {}

        print("Starting grid search optimization...")
        print("=" * 50)

        for epochs in epochs_grid:
            for batch_size in batch_sizes:
                for opt in optimizers:
                    try:
                        # Create and train model
                        model = model_fn(X_train, optimizer=opt)
                        fit_model(model, X_train, y_train, epochs, batch_size, validation_data, patience, verbose=0)
                        
                        # Generate predictions
                        yhat = prediction(model, X_test)
                        y_true_inv, yhat_inv = inverse_transform(y_test, yhat, scaler)
                        
                        # Calculate MAPE
                        mape = tf.keras.losses.MeanAbsolutePercentageError()(y_true_inv, yhat_inv).numpy()
                        print(f"MAPE: {mape:.4f}  | Optimizer: {opt}, Epochs: {epochs}, Batch Size: {batch_size}")

                        # Update best configuration
                        if mape < best_mape:
                            best_mape = mape
                            best_config = {'optimizer': opt, 'epochs': epochs, 'batch_size': batch_size}
                            
                    except Exception as model_error:
                        print(f"Error with config Optimizer: {opt}, Epochs: {epochs}, Batch Size: {batch_size}: {str(model_error)}")
                        continue

        print("\n" + "=" * 50)
        print("Best Config:", best_config, "with MAPE:", best_mape)
        return best_config
        
    except Exception as e:
        print(f"Error during grid search: {str(e)}")
        return {}