import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class BanknoteMLP(nn.Module):
    def __init__(self, input_size=4, hidden1=16, hidden2=8):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )

    def forward(self, x):
        return self.model(x)


def train_pytorch_model(X_train, y_train, X_val, y_val, epochs=500, lr=0.001):
    torch.manual_seed(42)
    np.random.seed(42)

    model = BanknoteMLP(input_size=X_train.shape[1], hidden1=16, hidden2=8)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values.reshape(-1, 1), dtype=torch.float32)

    history = {
        "train_loss": [],
        "val_loss": []
    }

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t)

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss.item())

        if epoch % 50 == 0:
            print(
                f"epoch {epoch}, "
                f"train_loss: {loss.item():.6f}, "
                f"val_loss: {val_loss.item():.6f}"
            )

    return model, history


def evaluate_pytorch_model(model, X, y):
    model.eval()

    X_t = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_t)
        probs = torch.sigmoid(logits).numpy().flatten()

    y_pred = (probs > 0.5).astype(int)
    y_true = y.to_numpy()

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }

    return results