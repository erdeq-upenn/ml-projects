import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
)


class _Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def run(X_train, X_test, y_train, y_test):
    """Returns (metrics_dict, predict_fn) where predict_fn(X) -> (preds, probs)."""
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"  [PyTorch] using device: {device}")

    # Validation split: 10% of train data
    n_total = X_train.shape[0]
    n_val = max(1, int(n_total * 0.1))
    n_train = n_total - n_val

    rng = np.random.default_rng(42)
    indices = rng.permutation(n_total)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    train_X = X_train[train_idx]
    train_y = y_train[train_idx]
    val_X = X_train[val_idx]
    val_y = y_train[val_idx]

    train_dataset = TensorDataset(torch.tensor(train_X), torch.tensor(train_y))
    loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)

    val_X_t = torch.tensor(val_X).to(device)
    val_y_t = torch.tensor(val_y).to(device)

    model = _Net(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5
    )

    max_epochs = 150
    patience = 15
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_weights = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_X_t), val_y_t).item()

        scheduler.step(val_loss)

        print(f"  Epoch {epoch}/{max_epochs} — loss: {avg_loss:.4f} | val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()

    def predict_fn(X):
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.tensor(X).to(device))).cpu().numpy()
        return (probs > 0.5).astype(int), probs

    y_pred, y_prob = predict_fn(X_test)

    metrics = {
        "Model": "PyTorch (Neural Net)",
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
    }

    return metrics, predict_fn