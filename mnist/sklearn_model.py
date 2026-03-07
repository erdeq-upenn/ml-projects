from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
)


def run(X_train, X_test, y_train, y_test):
    """Returns (metrics_dict, predict_fn) where predict_fn(X) -> (preds, probs)."""
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        max_iter=50,
        random_state=42,
        verbose=True,
        early_stopping=True,
        n_iter_no_change=10,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)  # (N, 10)

    metrics = {
        "Model": "Sklearn (MLP)",
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred, average="weighted"),
        "AUC-ROC": roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
    }

    def predict_fn(X):
        probs = model.predict_proba(X)
        return model.predict(X), probs

    return metrics, predict_fn
