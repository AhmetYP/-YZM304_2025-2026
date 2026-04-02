from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def train_sklearn_mlp(X_train, y_train, hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42):
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="sgd",
        learning_rate_init=0.01,
        max_iter=max_iter,
        random_state=random_state
    )

    model.fit(X_train, y_train)
    return model


def evaluate_sklearn_model(model, X, y):
    y_pred = model.predict(X)

    results = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y, y_pred)
    }

    return results