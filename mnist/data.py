import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from typing import NamedTuple
from sklearn.preprocessing import StandardScaler


class DataSplit(NamedTuple):
    X_train: np.ndarray
    X_test: np.ndarray
    X_train_scaled: np.ndarray
    X_test_scaled: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


def load_mnist() -> DataSplit:
    import ssl
    import tensorflow as tf

    # Scoped SSL bypass for the download only (corporate proxy with self-signed cert)
    _orig_ctx = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    finally:
        ssl._create_default_https_context = _orig_ctx

    X_train = X_train.reshape(-1, 784).astype(np.float32) / 255.0
    X_test = X_test.reshape(-1, 784).astype(np.float32) / 255.0
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    return DataSplit(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler)
