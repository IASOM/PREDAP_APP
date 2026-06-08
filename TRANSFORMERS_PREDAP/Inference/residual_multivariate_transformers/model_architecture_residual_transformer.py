"""Residual multivariate transformer model used by the inference bundle."""

import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam

from config.base_transformer_config import BaseTransformerConfig
from model_architechture.transformer_univ_architechtures.model_architechture_informer import build_informer_model
from model_architechture.transformer_univ_architechtures.model_architechture_log_transformer import build_log_transformer_model
from model_architechture.transformer_univ_architechtures.model_architechture_LSTNet import build_lstnet_model
from model_architechture.transformer_univ_architechtures.model_architechture_base_tranformer import build_base_model


def transformer_encoder(inputs, head_size=None, num_heads=None, ff_dim=None, dropout=None, activation_function="tanh"):
    default_config = BaseTransformerConfig()
    if head_size is None:
        head_size = default_config.DEFAULT_RESIDUAL_TRANSFORMER_PARAMS["head_size"]
    if num_heads is None:
        num_heads = default_config.DEFAULT_RESIDUAL_TRANSFORMER_PARAMS["num_heads"]
    if ff_dim is None:
        ff_dim = default_config.DEFAULT_RESIDUAL_TRANSFORMER_PARAMS["ff_dim"]
    if dropout is None:
        dropout = default_config.DEFAULT_RESIDUAL_TRANSFORMER_PARAMS["dropout"]

    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation=activation_function)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


@keras.utils.register_keras_serializable(package="predap")
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


@keras.utils.register_keras_serializable(package="predap")
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


def hybrid_lstm_transformer_model(input_shape, forecast, lstm_params=None, transformer_params=None, activation_function="tanh"):
    default_config = BaseTransformerConfig()
    if lstm_params is None:
        lstm_params = default_config.DEFAULT_RESIDUAL_LSTM_PARAMS.copy()
    if transformer_params is None:
        transformer_params = default_config.DEFAULT_RESIDUAL_TRANSFORMER_PARAMS.copy()

    input_layer = keras.Input(shape=input_shape)
    revin_layer = RevIN()
    x, mean, stdev = revin_layer(input_layer, mode="norm")

    d_model = max(transformer_params["head_size"] * transformer_params["num_heads"], 32)
    x = layers.Dense(d_model)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = PositionalEncoding(input_shape[0], d_model)(x)

    for _ in range(transformer_params["num_transformer_blocks"]):
        x = transformer_encoder(
            x,
            head_size=transformer_params["head_size"],
            num_heads=transformer_params["num_heads"],
            ff_dim=transformer_params["ff_dim"],
            activation_function=activation_function,
            dropout=transformer_params["dropout"],
        )

    if input_shape[0] >= 60:
        x = layers.AveragePooling1D(30, data_format="channels_first")(x)
    x = layers.Flatten()(x)
    for dim in transformer_params["mlp_units"]:
        x = layers.Dense(dim, activation=activation_function)(x)
        x = layers.Dropout(transformer_params["dropout"])(x)
    outputs = layers.Dense(forecast, activation="linear")(x)
    outputs = layers.Reshape((forecast, 1))(outputs)
    outputs = revin_layer(outputs, mode="denorm", mean=mean, stdev=stdev)
    outputs = layers.Reshape((forecast,))(outputs)
    return keras.Model(inputs=input_layer, outputs=outputs)


class CustomCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr=1e-4, max_lr=1e-3, min_lr=1e-5, warmup_steps=20, total_steps=50):
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

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
        }


@keras.utils.register_keras_serializable(package="predap")
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


@keras.utils.register_keras_serializable(package="predap")
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
