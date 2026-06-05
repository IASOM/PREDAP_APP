import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def get_log_sparse_mask(seq_len):
    col_indices = tf.range(seq_len)[tf.newaxis, :]
    row_indices = tf.range(seq_len)[:, tf.newaxis]
    distance = row_indices - col_indices
    local_mask = tf.logical_or(tf.equal(distance, 0), tf.equal(distance, 1))
    dist_f = tf.cast(distance, tf.float32)
    log2_dist = tf.math.log(tf.maximum(dist_f, 1.0)) / tf.math.log(2.0)
    log_mask = tf.equal(tf.math.abs(log2_dist - tf.math.round(log2_dist)), 0.0)
    log_mask = tf.logical_and(tf.greater(distance, 0), log_mask)
    return tf.logical_or(local_mask, log_mask)


def log_transformer_encoder(inputs, head_size, num_heads, ff_dim, activation_function="tanh", dropout=0):
    seq_len = tf.shape(inputs)[1]
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    log_mask = get_log_sparse_mask(seq_len)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(query=x, value=x, attention_mask=log_mask)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation=activation_function)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def transformer_encoder(inputs, head_size, num_heads, ff_dim, activation_function="tanh", dropout=0, causal_masking=False):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation=activation_function)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_log_transformer_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, activation_function="tanh", dropout=0, mlp_dropout=0, n_pred=1, pos_encoding=False):
    inputs = keras.Input(shape=input_shape)
    revin_layer = RevIN()
    x, mean, stdev = revin_layer(inputs, mode="norm")
    d_model = max(head_size * num_heads, 32)
    x = layers.Dense(d_model)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = PositionalEncoding(input_shape[0], d_model)(x)
    for _ in range(num_transformer_blocks):
        x = log_transformer_encoder(x, head_size, num_heads, ff_dim, activation_function, dropout)
    x = layers.Flatten()(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation=activation_function)(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_pred)(x)
    outputs = layers.Reshape((n_pred, 1))(outputs)
    outputs = revin_layer(outputs, mode="denorm", mean=mean, stdev=stdev)
    outputs = layers.Reshape((n_pred,))(outputs)
    return keras.Model(inputs, outputs)


class CustomCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr=1e-5, max_lr=5e-5, min_lr=1e-6, warmup_steps=20, total_steps=50):
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        if step < self.warmup_steps:
            return self.initial_lr + (self.max_lr - self.initial_lr) * (step / self.warmup_steps)
        decay_steps = self.total_steps - self.warmup_steps
        step_after_warmup = step - self.warmup_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step_after_warmup / decay_steps))
        return (self.max_lr - self.min_lr) * cosine_decay + self.min_lr


@keras.saving.register_keras_serializable(package="predap")
class PositionalEncoding(layers.Layer):
    def __init__(self, sequence_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        pos = tf.range(start=0, limit=sequence_length, delta=1, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(start=0, limit=d_model, delta=1, dtype=tf.float32)[tf.newaxis, :]
        angle_rates = 1 / (10000 ** ((2 * (i // 2)) / tf.cast(d_model, tf.float32)))
        angle_rads = pos * angle_rates
        sines = tf.sin(angle_rads[:, 0::2])
        coses = tf.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, coses], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        self.pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]


@keras.saving.register_keras_serializable(package="predap")
class RevIN(layers.Layer):
    def __init__(self, eps=1e-5, detach_grad=False, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.detach_grad = detach_grad

    def call(self, x, mode="norm", mean=None, stdev=None):
        if mode == "norm":
            mean, stdev = self._get_statistics(x)
            return (x - mean) / stdev, mean, stdev
        if mode == "denorm":
            if mean is None or stdev is None:
                raise ValueError("For mode='denorm', mean and stdev must be provided.")
            dims = tf.shape(x)[-1]
            return x * stdev[:, :, :dims] + mean[:, :, :dims]
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
        config.update({"eps": self.eps, "detach_grad": self.detach_grad})
        return config
