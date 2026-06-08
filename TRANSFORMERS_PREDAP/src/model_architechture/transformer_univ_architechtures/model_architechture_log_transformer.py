import tensorflow as tf
import math
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam


def get_log_sparse_mask(seq_len):
    """
    Creates a LogSparse boolean mask using pure TensorFlow operations.
    Works with KerasTensors.
    """
    # Create a grid of indices: (seq_len, seq_len)
    col_indices = tf.range(seq_len)[tf.newaxis, :]
    row_indices = tf.range(seq_len)[:, tf.newaxis]
    
    # Distance matrix (how far back each cell is looking)
    # Only look at the past (row >= col)
    distance = row_indices - col_indices
    
    # 1. Local Mask: distance is 0 (self) or 1 (immediate neighbor)
    local_mask = tf.logical_or(tf.equal(distance, 0), tf.equal(distance, 1))
    
    # 2. Log Mask: distance is a power of 2 (2^0, 2^1, 2^2...)
    # We check if (log2(distance)) is an integer
    # Note: distance must be > 0 for log2
    dist_f = tf.cast(distance, tf.float32)
    log2_dist = tf.math.log(tf.maximum(dist_f, 1.0)) / tf.math.log(2.0)
    
    # Check if the log2 value is very close to an integer (accounting for float precision)
    log_mask = tf.equal(tf.math.abs(log2_dist - tf.math.round(log2_dist)), 0.0)
    # Ensure we only apply log mask to positive distances
    log_mask = tf.logical_and(tf.greater(distance, 0), log_mask)
    
    # Combine: Local + Logarithmic
    full_mask = tf.logical_or(local_mask, log_mask)
    
    # Optional: If you want to strictly prevent looking into the future (Causal)
    # the 'distance >= 0' logic above already handles this.
    
    return full_mask

def log_transformer_encoder(inputs, head_size, num_heads, ff_dim, activation_function='tanh', dropout=0):
    # inputs shape: (batch, seq_len, features)
    seq_len = tf.shape(inputs)[1]
    
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    
    # This now receives a Tensor and returns a Tensor safely
    log_mask = get_log_sparse_mask(seq_len)

    # MultiHeadAttention
    x = layers.MultiHeadAttention(
        key_dim=head_size, 
        num_heads=num_heads, 
        dropout=dropout
    )(query=x, value=x, attention_mask=log_mask) 
    
    x = layers.Dropout(dropout)(x)
    res = x + inputs 

    # Feed Forward
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation=activation_function)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    
    return x + res


def transformer_encoder(inputs, head_size, num_heads, ff_dim, activation_function='tanh', dropout=0, causal_masking=False):
    """
    Transformer encoder block with multi-head attention and feed-forward layers.
    
    Args:
        inputs: Input tensor (batch_size, sequence_length, features)
        head_size: Dimension of each attention head
        num_heads: Number of attention heads
        ff_dim: Dimension of feed-forward layer
        dropout: Dropout rate
        
    Returns:
        Encoded tensor with residual connections
    """

    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)  # to inputs to stabilize training
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)  # self attention to normalized input 
    x = layers.Dropout(dropout)(x)  # (dropout to reduce overfitting)
    res = x + inputs  # attention output added to original inputs
    

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)  # again after resid connection
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation=activation_function)(x)  # point-wise convol.: Expands feature dim to ff_dim 
    x = layers.Dropout(dropout)(x)  # dropout again
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)  # reduces feature dim back to match input size
    x = x + res
    
    return x


def build_log_transformer_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, activation_function = "tanh", dropout=0, mlp_dropout=0, n_pred=1, pos_encoding = False):
    """
    Build complete transformer model for univariate time series forecasting.
    
    Args:
        input_shape: Shape of input data (sequence_length, features)
        head_size: Dimension of each attention head
        num_heads: Number of attention heads
        ff_dim: Dimension of feed-forward layer
        num_transformer_blocks: Number of transformer encoder blocks
        mlp_units: List of MLP layer dimensions
        dropout: Dropout rate for transformer layers
        mlp_dropout: Dropout rate for MLP layers
        n_pred: Number of prediction steps
        
    Returns:
        Compiled Keras model
    """
    inputs = keras.Input(shape=input_shape)
    
    # RevIN and Initial Projection
    revin_layer = RevIN() # Assuming your RevIN class is defined elsewhere
    x, mean, stdev = revin_layer(inputs, mode='norm')
    
    d_model = max(head_size * num_heads, 32)
    x = layers.Dense(d_model)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Add Positional Encoding (Crucial for LogTransformer to know 'distance')
    x = PositionalEncoding(input_shape[0], d_model)(x) 
    
    # Apply LogTransformer Blocks
    for _ in range(num_transformer_blocks):
        x = log_transformer_encoder(x, head_size, num_heads, ff_dim, activation_function, dropout)

    # Final MLP Head
    x = layers.Flatten()(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation=activation_function)(x)
        x = layers.Dropout(mlp_dropout)(x)
   
    outputs = layers.Dense(n_pred)(x)
    # De-normalization
    outputs = layers.Reshape((n_pred, 1))(outputs)
    outputs = revin_layer(outputs, mode='denorm', mean=mean, stdev=stdev)
    outputs = layers.Reshape((n_pred,))(outputs)

    return keras.Model(inputs, outputs)


class CustomCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom cosine decay learning rate schedule with warmup phase.
    Defaults to 50 epochs total and 20 warmup steps.
    """
    
    def __init__(self, initial_lr=1e-5, max_lr=5e-5, min_lr=1e-6, warmup_steps=20, total_steps=50):
        """
        Initialize the learning rate schedule.
        
        Args:
            initial_lr: Starting learning rate (0.01)
            max_lr: Maximum learning rate (0.1)
            min_lr: Final learning rate (0.0001)
            warmup_steps: Number of warmup steps to reach max_lr
            total_steps: Total number of steps (decay after warmup)
        """
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        """Apply the learning rate schedule."""
        # Warm-up phase: linearly increase to max_lr
        if step < self.warmup_steps:
            return self.initial_lr + (self.max_lr - self.initial_lr) * (step / self.warmup_steps)

        # Cosine decay phase after warmup
        decay_steps = self.total_steps - self.warmup_steps
        step_after_warmup = step - self.warmup_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step_after_warmup / decay_steps))
        decayed = (self.max_lr - self.min_lr) * cosine_decay + self.min_lr
        return decayed

@keras.utils.register_keras_serializable(package="predap")
class PositionalEncoding(layers.Layer):
    def __init__(self, sequence_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model

        # Create the positional encoding matrix once
        pos = tf.range(start=0, limit=sequence_length, delta=1, dtype=tf.float32)[:, tf.newaxis]  # (seq_len, 1)
        i = tf.range(start=0, limit=d_model, delta=1, dtype=tf.float32)[tf.newaxis, :]             # (1, d_model)
        # compute the angle rates
        angle_rates = 1 / (10000 ** ( (2 * (i//2)) / tf.cast(d_model, tf.float32) ))                  # (1, d_model)
        angle_rads = pos * angle_rates                                                              # (seq_len, d_model)

        # apply sin to even indices in the array; cos to odd indices
        sines = tf.sin(angle_rads[:, 0::2])
        coses = tf.cos(angle_rads[:, 1::2])
        # now interleave sines & coses into one matrix
        pos_encoding = tf.concat([sines, coses], axis=-1)                                           # (seq_len, d_model)
        pos_encoding = pos_encoding[tf.newaxis, ...]                                                # (1, seq_len, d_model)
        self.pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]

@keras.utils.register_keras_serializable(package="predap")
class RevIN(layers.Layer):
    def __init__(self, eps=1e-5, detach_grad=False, **kwargs):
        super(RevIN, self).__init__(**kwargs)
        self.eps = eps
        self.detach_grad = detach_grad

    def call(self, x, mode='norm', mean=None, stdev=None):
        if mode == 'norm':
            mean, stdev = self._get_statistics(x)
            x_norm = (x - mean) / stdev
            return x_norm, mean, stdev

        elif mode == 'denorm':
            if mean is None or stdev is None:
                raise ValueError("For mode='denorm', mean and stdev must be provided.")
            dims = tf.shape(x)[-1]
            return x * stdev[:, :, :dims] + mean[:, :, :dims]

        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def _get_statistics(self, x):
        mean = tf.reduce_mean(x, axis=1, keepdims=True)
        stdev = tf.math.reduce_std(x, axis=1, keepdims=True) + self.eps

        if self.detach_grad:
            mean = tf.stop_gradient(mean)
            stdev = tf.stop_gradient(stdev)

        return mean, stdev

    def get_config(self):
        config = super().get_config()
        config.update({
            "eps": self.eps,
            "detach_grad": self.detach_grad,
        })
        return config
