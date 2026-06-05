import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable()
class LSTNetSkipBlock(layers.Layer):
    def __init__(self, skip, skip_units, **kwargs):
        super().__init__(**kwargs)
        self.skip = skip
        self.skip_units = skip_units

    def build(self, input_shape):
        self.n_filters = input_shape[-1]
        self.gru = layers.GRU(self.skip_units, recurrent_activation="relu", name="skip_gru")
        super().build(input_shape)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        num_skips = seq_len // self.skip

        def process_skip():
            x_sliced = x[:, -num_skips * self.skip :, :]
            x_reshaped = tf.reshape(x_sliced, [batch_size, num_skips, self.skip, self.n_filters])
            x_transposed = tf.transpose(x_reshaped, [0, 2, 1, 3])
            x_folded = tf.reshape(x_transposed, [-1, num_skips, self.n_filters])
            s_out = self.gru(x_folded)
            return tf.reshape(s_out, [batch_size, self.skip * self.skip_units])

        def process_empty():
            return tf.zeros([batch_size, self.skip * self.skip_units])

        return tf.cond(num_skips > 0, process_skip, process_empty)

    def get_config(self):
        config = super().get_config()
        config.update({"skip": self.skip, "skip_units": self.skip_units})
        return config


def build_lstnet_model(input_shape, n_filters=100, kernel_size=6, rnn_units=100, skip_units=5, skip=24, n_pred=1, dropout=0.2):
    inputs = keras.Input(shape=input_shape)
    revin_layer = RevIN()
    x_norm, mean, stdev = revin_layer(inputs, mode="norm")
    c = layers.Conv1D(filters=n_filters, kernel_size=kernel_size, activation="relu", padding="same")(x_norm)
    c = layers.Dropout(dropout)(c)
    r = layers.GRU(rnn_units, recurrent_activation="relu")(c)
    s_path = LSTNetSkipBlock(skip=skip, skip_units=skip_units)(c)
    z = layers.Flatten()(x_norm)
    res_linear = layers.Dense(n_pred)(z)
    combined = layers.Concatenate()([r, s_path])
    res_nonlinear = layers.Dense(n_pred)(combined)
    main_output = layers.Add()([res_nonlinear, res_linear])
    main_output = layers.Reshape((n_pred, 1))(main_output)
    main_output = revin_layer(main_output, mode="denorm", mean=mean, stdev=stdev)
    main_output = layers.Reshape((n_pred,))(main_output)
    return keras.Model(inputs, main_output)


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
