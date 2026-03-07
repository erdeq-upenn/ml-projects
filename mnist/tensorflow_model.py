import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
)
from sklearn.preprocessing import StandardScaler

_DEFAULT_ARTIFACT_DIR = Path(__file__).parent.parent / "models" / "mnist" / "tensorflow"


def train_and_save(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler: StandardScaler,
    artifact_dir: Path = _DEFAULT_ARTIFACT_DIR,
) -> dict:
    """Train model, save SavedModel + scaler to disk, return metrics."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    tf.random.set_seed(42)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1, verbose=1, callbacks=[es])

    model.save(artifact_dir / "model")
    joblib.dump(scaler, artifact_dir / "scaler.joblib")

    y_prob = model.predict(X_test, verbose=0)  # (N, 10)
    y_pred = np.argmax(y_prob, axis=1)

    return {
        "Model": "TensorFlow (Neural Net)",
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred, average="weighted"),
        "AUC-ROC": roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
    }


def load_and_predict(artifact_dir: Path = _DEFAULT_ARTIFACT_DIR):
    """Load SavedModel from disk, return predict_fn(X) -> (preds, probs)."""
    model = tf.keras.models.load_model(artifact_dir / "model")

    def predict_fn(X):
        probs = model.predict(X, verbose=0)
        return np.argmax(probs, axis=1), probs

    return predict_fn