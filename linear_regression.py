import numpy as np
import pandas as pd

# [w_0, w_1]
weights = np.zeros(2)

def predict(x: np.ndarray) -> float:
    return weights[0] * x[0] + weights[1] * x[1]

def feedForward(X: np.ndarray) -> np.ndarray:
    y_hats: np.ndarray = np.array([])
    for x in X:
        y_hat = predict(x)
        y_hats = np.append(y_hats, y_hat)
    return y_hats

def fit(X: np.ndarray, Y: np.ndarray):
    assert X.shape[0] == Y.shape[0]
    sample_size = X.shape[0]

    deltas = np.array([])
    for j in range(len(weights)):
        delta = np.sum([((predict(X[i]) - Y[i]) * X[i][j]) for i in range(sample_size)]) / sample_size
        deltas = np.append(deltas, delta)

    for i in range(len(weights)):
        alpha = 0.00003
        weights[i] -= alpha * deltas[i]

def error(y_hats: np.ndarray, y: np.ndarray):
    sample_size = y.shape[0]
    return np.sum([((y_hats[i] - y[i]) ** 2) for i in range(sample_size)]) / (2 * sample_size)


x = np.array([
    np.array([1, 100]),
    np.array([1, 150]),
    np.array([1, 200]),
])
y = np.array([
    200,
    280,
    405,
])

y_hats = feedForward(x)
print(y_hats)
print(error(y_hats, y))

fit(x, y)

y_hats = feedForward(x)
print(y_hats)
print(error(y_hats, y))

fit(x, y)

y_hats = feedForward(x)
print(y_hats)
print(error(y_hats, y))

fit(x, y)

y_hats = feedForward(x)
print(y_hats)
print(error(y_hats, y))

fit(x, y)

y_hats = feedForward(x)
print(y_hats)
print(error(y_hats, y))

fit(x, y)

y_hats = feedForward(x)
print(y_hats)
print(error(y_hats, y))

fit(x, y)

y_hats = feedForward(x)
print(y_hats)
print(error(y_hats, y))

fit(x, y)

y_hats = feedForward(x)
print(y_hats)
print(error(y_hats, y))

fit(x, y)

y_hats = feedForward(x)
print(y_hats)
print(error(y_hats, y))

fit(x, y)

y_hats = feedForward(x)
print(y_hats)
print(error(y_hats, y))

y_hats = feedForward([
    np.array([1, 10]),
    np.array([1, 100]),
    np.array([1, 500]),
])
print(y_hats)