import numpy as np

train_X = np.array([
    np.array([0, 0]),
    np.array([0, 1]),
    np.array([1, 0]),
    np.array([1, 1]),
    np.array([1, 1]),
    np.array([1, 1]),
])
train_y = np.array([
    np.array([0]),
    np.array([0]),
    np.array([0]),
    np.array([1]),
    np.array([1]),
    np.array([1]),
])

weights_h = np.array([
    np.array([0.5, 0.5]),
    np.array([0.5, 0.5]),
    np.array([0.5, 0.5]),
])
biases_h = np.array([
    0,
    0,
    0,
])
weights_out = np.array([0.5, 0.5, 0.5])
bias_out = 0

def sigmoid(z: np.ndarray) -> float:
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(x: np.ndarray) -> float:
    return sigmoid(x) * (1 - sigmoid(x))

leaky_relu_alpha = 0.01
def leaky_relu(z: np.ndarray) -> float:
    return np.where(z > 0, z, leaky_relu_alpha * z)

def leaky_relu_derivative(x: np.ndarray) -> float:
    return np.where(x > 0, 1, leaky_relu_alpha)

def squareError(y_hat: float, y: float) -> float:
    return (y_hat - y) ** 2

def forwardPass(X: np.ndarray) -> float:
    z_h = X @ weights_h.T + biases_h
    # a_h = sigmoid(z_h)
    a_h = leaky_relu(z_h)

    z_o = a_h @ weights_out.T + bias_out
    # a_o = sigmoid(z_o)
    a_o = leaky_relu(z_o)

    return a_o

prev_o_dmse_dw = np.zeros(weights_out.shape)
prev_o_dmse_db = 0
prev_gradients = [[np.zeros(weights_h.shape[1]), 0] for i in range(len(weights_h))]

def backwardPass(Xs: np.ndarray, ys: float):
    global weights_out
    global bias_out
    global weights_h
    global biases_h
    global prev_o_dmse_dw
    global prev_o_dmse_db
    global prev_gradients

    sample_count = len(Xs)

    o_dmse_dw = np.zeros(weights_out.shape)
    o_dmse_db = 0

    gradients = [[np.zeros(weights_h.shape[1]), 0] for i in range(len(weights_h))]

    for (X, y) in zip(Xs, ys):
        y = y[0]
        z_h = X @ weights_h.T + biases_h
        # a_h = sigmoid(z_h)
        a_h = leaky_relu(z_h)

        z_o = a_h @ weights_out.T + bias_out
        # a_o = sigmoid(z_o)
        a_o = leaky_relu(z_o)

        # print(f"X: {X}")
        # print(f"a_h: {a_h}")
        # print(f"y_hat: {a_o}, y: {y}")

        o_dmse_dy_hat = (a_o - y) * 2
        # o_dy_hat_dz = sigmoid_derivative(z_o)
        o_dy_hat_dz = leaky_relu_derivative(z_o)
        o_dz_dw = a_h

        o_dmse_dz = o_dmse_dy_hat * o_dy_hat_dz
        o_dmse_dw += o_dmse_dz * o_dz_dw / sample_count
        o_dmse_db += o_dmse_dz.sum() / sample_count

        for i in range(len(weights_h)):
            h_dzo_da = weights_out[i]
            # h_da_dz = sigmoid_derivative(z_h[i])
            h_da_dz = leaky_relu_derivative(z_h[i])
            h_dz_dw = X

            h_dzo_dz = h_dzo_da * h_da_dz

            h_dzo_dw = h_dzo_dz * h_dz_dw
            h_dmse_dw = o_dmse_dz * h_dzo_dw
            h_dzo_db = h_dzo_dz.sum()
            h_dmse_db = o_dmse_dz * h_dzo_db

            gradients[i][0] += h_dmse_dw
            gradients[i][1] += h_dmse_db

    # print(f"o_dmse_dw: {o_dmse_dw}")
    # print(f"o_dmse_db: {o_dmse_db}")
    # for gradient in gradients:
    #     print(f"gradient: {gradient}")
    # print()

    alpha = 0.01
    beta = 0
    # Update weights and biases
    weights_out += alpha * -o_dmse_dw + beta * prev_o_dmse_dw
    bias_out += alpha * -o_dmse_db + beta * prev_o_dmse_db
    for i in range(len(weights_h)):
        weights_h[i] += alpha * -gradients[i][0] + beta * prev_gradients[i][0]
        biases_h[i] += alpha * -gradients[i][1] + beta * prev_gradients[i][1]

    prev_o_dmse_dw = o_dmse_dw
    prev_o_dmse_db = o_dmse_db
    prev_gradients = gradients

y_hat = forwardPass(train_X[3])
print(f"initial y_hat: {y_hat} y: {train_y[3][0]}")
print(f"initial error: {squareError(y_hat, train_y[3][0])}")
print()

for i in range(1000):
    backwardPass(train_X, train_y)

    if i % 100 == 0:
        # print(f"y_hat: {y_hat} y: {train_y[3][0]}")
        # print(f"error: {squareError(y_hat, train_y[3][0])}")
        squareErrors = 0
        for (X, y) in zip(train_X, train_y):
            y_hat = forwardPass(train_X[3])
            squareErrors += squareError(y_hat, train_y[3][0])
        print(f"MSE: {squareErrors / len(train_X)}")

y_hat = forwardPass(train_X[3])
print(f"y_hat: {y_hat} y: {train_y[3][0]}")
y_hat = forwardPass(train_X[1])
print(f"y_hat: {y_hat} y: {train_y[1][0]}")
y_hat = forwardPass(train_X[0])
print(f"y_hat: {y_hat} y: {train_y[0][0]}")