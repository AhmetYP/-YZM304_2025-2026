import numpy as np


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def relu(Z):
    return np.maximum(0, Z)


def relu_derivative(Z):
    return (Z > 0).astype(float)


def initialize_parameters(n_x, n_h, n_y=1):
    np.random.seed(42)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))

    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    return {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }


def forward_propagation(X, parameters):
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]

    Z1 = np.dot(W1, X)
    Z1 = Z1 + b1
    A1 = sigmoid(Z1)

    Z2 = np.dot(W2, A1)
    Z2 = Z2 + b2
    A2 = sigmoid(Z2)

    cache = {
        "A1": A1,
        "A2": A2
    }

    return A2, cache


def compute_cost(A2, Y):
    m = Y.shape[1]
    eps = 1e-8

    cost = -(1 / m) * np.sum(
        Y * np.log(A2 + eps) + (1 - Y) * np.log(1 - A2 + eps)
    )

    return float(cost)


def backpropagation(X, Y, parameters, cache):
    m = X.shape[1]

    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }


def update_parameters(parameters, grads, learning_rate=0.1):
    parameters["W1"] -= learning_rate * grads["dW1"]
    parameters["b1"] -= learning_rate * grads["db1"]
    parameters["W2"] -= learning_rate * grads["dW2"]
    parameters["b2"] -= learning_rate * grads["db2"]

    return parameters


def train_model(X, Y, X_val=None, Y_val=None, n_h=6, n_steps=1000, lr=0.1):
    n_x = X.shape[0]
    parameters = initialize_parameters(n_x, n_h)

    history = {
        "train_loss": [],
        "val_loss": []
    }

    for i in range(n_steps):
        A2_train, cache = forward_propagation(X, parameters)
        train_cost = compute_cost(A2_train, Y)
        grads = backpropagation(X, Y, parameters, cache)
        parameters = update_parameters(parameters, grads, lr)

        history["train_loss"].append(train_cost)

        if X_val is not None and Y_val is not None:
            A2_val, _ = forward_propagation(X_val, parameters)
            val_cost = compute_cost(A2_val, Y_val)
            history["val_loss"].append(val_cost)

        if i % 100 == 0:
            if X_val is not None and Y_val is not None:
                print(f"step {i}, train_cost: {train_cost:.6f}, val_cost: {val_cost:.6f}")
            else:
                print(f"step {i}, cost: {train_cost:.6f}")

    return parameters, history


def predict(parameters, X):
    A2, _ = forward_propagation(X, parameters)
    return (A2 > 0.5).astype(int)

def initialize_parameters_2hidden(n_x, n_h1, n_h2, n_y=1):
    np.random.seed(42)

    W1 = np.random.randn(n_h1, n_x) * np.sqrt(2 / n_x)
    b1 = np.zeros((n_h1, 1))

    W2 = np.random.randn(n_h2, n_h1) * np.sqrt(2 / n_h1)
    b2 = np.zeros((n_h2, 1))

    W3 = np.random.randn(n_y, n_h2) * 0.01
    b3 = np.zeros((n_y, 1))

    return {
        "W1": W1, "b1": b1,
        "W2": W2, "b2": b2,
        "W3": W3, "b3": b3
    }

def forward_propagation_2hidden(X, parameters):
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    W3, b3 = parameters["W3"], parameters["b3"]

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)

    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = {
        "Z1": Z1, "A1": A1,
        "Z2": Z2, "A2": A2,
        "A3": A3
    }

    return A3, cache

def backpropagation_2hidden(X, Y, parameters, cache):
    m = X.shape[1]

    W2 = parameters["W2"]
    W3 = parameters["W3"]

    Z1, A1 = cache["Z1"], cache["A1"]
    Z2, A2 = cache["Z2"], cache["A2"]
    A3 = cache["A3"]

    dZ3 = A3 - Y
    dW3 = (1 / m) * np.dot(dZ3, A2.T)
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

    dZ2 = np.dot(W3.T, dZ3) * relu_derivative(Z2)
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2) * relu_derivative(Z1)
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return {
        "dW1": dW1, "db1": db1,
        "dW2": dW2, "db2": db2,
        "dW3": dW3, "db3": db3
    }

def update_parameters_2hidden(parameters, grads, learning_rate=0.1):
    parameters["W1"] -= learning_rate * grads["dW1"]
    parameters["b1"] -= learning_rate * grads["db1"]

    parameters["W2"] -= learning_rate * grads["dW2"]
    parameters["b2"] -= learning_rate * grads["db2"]

    parameters["W3"] -= learning_rate * grads["dW3"]
    parameters["b3"] -= learning_rate * grads["db3"]

    return parameters


def train_model_2hidden(X, Y, X_val=None, Y_val=None, n_h1=16, n_h2=8, n_steps=1000, lr=0.1):
    n_x = X.shape[0]
    parameters = initialize_parameters_2hidden(n_x, n_h1, n_h2)

    history = {
        "train_loss": [],
        "val_loss": []
    }

    for i in range(n_steps):
        A3_train, cache = forward_propagation_2hidden(X, parameters)
        train_cost = compute_cost(A3_train, Y)
        grads = backpropagation_2hidden(X, Y, parameters, cache)
        parameters = update_parameters_2hidden(parameters, grads, lr)

        history["train_loss"].append(train_cost)

        if X_val is not None and Y_val is not None:
            A3_val, _ = forward_propagation_2hidden(X_val, parameters)
            val_cost = compute_cost(A3_val, Y_val)
            history["val_loss"].append(val_cost)

        if i % 100 == 0:
            if X_val is not None and Y_val is not None:
                print(f"step {i}, train_cost: {train_cost:.6f}, val_cost: {val_cost:.6f}")
            else:
                print(f"step {i}, cost: {train_cost:.6f}")

    return parameters, history


def predict_2hidden(parameters, X):
    A3, _ = forward_propagation_2hidden(X, parameters)
    return (A3 > 0.5).astype(int)