"""Utility helpers required by the inference bundle."""

from __future__ import annotations

import json
from pathlib import Path

import gc
import ctypes

import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import backend as K
import tensorflow as tf


_original_read_csv = pd.read_csv


def cleanup_ram():
    plt.close("all")
    K.clear_session()
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def smart_read(file_path, **kwargs):
    if str(file_path).lower().endswith(".parquet"):
        print(f"-> INFO: Leyendo {file_path} como PARQUET.")
        return pd.read_parquet(file_path, **kwargs)
    print(f"-> INFO: Leyendo {file_path} como CSV (o formato predeterminado).")
    return _original_read_csv(file_path, **kwargs)


# Override pandas read_csv globally so any direct pd.read_csv calls in the inference bundle also handle parquet.
pd.read_csv = smart_read


def load_json_codes_list(json_path: str) -> str:
    with open(json_path, "r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)
    return ",".join(data)


def memory_cleanup():
    K.clear_session()
    tf.compat.v1.reset_default_graph()
    gc.collect()
    plt.close("all")
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def compute_dynamic_batch_size(lookback, forecast):
    gpus = tf.config.list_physical_devices("GPU")
    if lookback <= 14 and forecast <= 14:
        batch_size = 4096
    elif lookback <= 30 and forecast <= 30:
        batch_size = 1024
    elif (30 <= lookback <= 60) and forecast <= 60:
        batch_size = 256
    elif 60 <= lookback <= 128 and forecast <= 128:
        batch_size = 128
    elif 128 < lookback <= 365 and forecast <= 365:
        batch_size = 128
    else:
        batch_size = 92

    if len(gpus) == 0:
        batch_size = 32

    return batch_size
