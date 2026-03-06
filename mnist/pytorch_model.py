import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
)


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def run(X_train, X_test, y_train, y_test):
    """Returns (metrics_dict, predict_fn) where predict_fn(X) -> (preds, probs)."""
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"  [PyTorch] using device: {device}")

    dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = _Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(10):
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"  Epoch {epoch + 1:02d}/10 — loss: {avg_loss:.4f}")

    model.eval()

    def predict_fn(X):
        with torch.no_grad():
            logits = model(torch.tensor(X).to(device))
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return np.argmax(probs, axis=1), probs

    y_pred, y_prob = predict_fn(X_test)

    metrics = {
        "Model": "PyTorch (Neural Net)",
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred, average="weighted"),
        "AUC-ROC": roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
    }

    return metrics, predict_fn
