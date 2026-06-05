import os

import tensorflow as tf
from pathlib import Path

def setup_gpu_memory():
    """
    Configure GPU memory growth to prevent TensorFlow from allocating all GPU memory.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU memory growth enabled for {gpu}")
        except RuntimeError as e:
            print(f"Error setting GPU memory growth: {e}")
    else:
        print("No GPUs found. Running on CPU.")

def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def create_model_directories():
    """
    Create necessary directories for model storage and plots.
    """
    directories = ['models', 'plots', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created/verified directory: {directory}")
