import numpy as np
from abc import ABC, abstractmethod

# ====== Activation funtion ====== #
class Activation(ABC):
    @abstractmethod
    def apply(self, x: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def apply_derivative(self, x: np.ndarray) -> np.ndarray:
        pass

class IdentityActivation(Activation):
    def apply(self, x: np.ndarray) -> np.ndarray:
        return x
    
    def apply_derivative(self, x: np.ndarray) -> np.ndarray:
        return 1
    
class SigmoidActivation(Activation):
    def apply(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def apply_derivative(self, x: np.ndarray) -> np.ndarray:
        return self.apply(x) * (1 - self.apply(x))
    
class LeakyReLUActivation(Activation):
    alpha: float

    def __init__(self, alpha: float):
        self.alpha = alpha

    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * x)
    
    def apply_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, self.alpha)

# ====== Optimizer function ====== #
class Optimizer(ABC):
    out_w: np.ndarray
    hidden_weights: list[np.ndarray]

    def set_weights_arrays(self, hidden_weights, out_w):
        self.hidden_weights = hidden_weights
        self.out_w = out_w

    @abstractmethod
    def update(self, hidden_deltas, out_dW):
        pass

class SGDOptimizer(Optimizer):
    lr: float

    def __init__(self, lr: float):
        super(Optimizer, self).__init__()
        self.lr = lr

    def update(self, hidden_deltas, out_dW):
        for i in range(len(hidden_deltas)):
            self.hidden_weights[i] += hidden_deltas[i] * self.lr
        self.out_w += out_dW * self.lr

class MomentumOptimizer(Optimizer):
    lr: float
    beta: float

    hidden_vs: list[np.ndarray]
    out_v: np.ndarray = 0

    def __init__(self, lr: float, beta: float):
        super(Optimizer, self).__init__()
        self.lr = lr
        self.beta = beta

    def set_weights_arrays(self, hidden_weights, out_w):
        self.hidden_vs = [0 for _ in range(len(hidden_weights))]
        Optimizer.set_weights_arrays(self, hidden_weights, out_w)

    def update(self, hidden_deltas, out_dW):
        for i in range(len(hidden_deltas)):
            self.hidden_vs = self.beta * self.hidden_vs[i] + (1 - self.beta) * hidden_deltas[i]
            self.hidden_weights[i] += self.hidden_vs[i] * self.lr

        self.out_v = self.beta * self.out_v + (1 - self.beta) * out_dW
        self.out_w += self.out_v * self.lr

class AdaGradOptimizer(Optimizer):
    lr: float

    Gs_hidden: list[np.ndarray]
    G_out: np.ndarray = 0

    def __init__(self, lr: float):
        super(Optimizer, self).__init__()
        self.lr = lr

    def set_weights_arrays(self, hidden_weights, out_w):
        self.Gs_hidden = [0 for i in range(len(hidden_weights))]
        Optimizer.set_weights_arrays(self, hidden_weights, out_w)

    def update(self, hidden_deltas, out_dW):
        for i in range(len(hidden_deltas)):
            self.Gs_hidden[i] += hidden_deltas[i] ** 2
            self.hidden_weights[i] += hidden_deltas[i] * self.lr / (0.00000001 + np.sqrt(self.Gs_hidden[i]))

        self.G_out += out_dW ** 2
        self.out_w += out_dW * self.lr / (0.00000001 + np.sqrt(self.G_out))

# Base classifier class
class Classifier(ABC):
    @abstractmethod
    def fit(self, X, y):
        # Abstract method to fit the model with features X and target y
        pass

    @abstractmethod
    def predict(self, X):
        # Abstract method to make predictions on the dataset X
        pass

    @abstractmethod
    def predict_proba(self, X):
        # Abstract method predict the probability of the dataset X
        pass


class MLPClassifier(Classifier):
    activation: Activation
    optimizer: Optimizer
    learning_rate: float
    n_epoch: int

    def __init__(self, layers: list[int], activation: Activation, optimizer: Optimizer, n_epoch = 10_000):
        self.layers = layers
        self.activation = activation
        self.optimizer = optimizer
        self.n_epoch = n_epoch

        self.hidden_weights = []
        for i in range(len(self.layers) - 2):
            self.hidden_weights.append(np.random.rand(self.layers[i], self.layers[i + 1]) * 0.1)

        self.weights_out = np.random.rand(self.layers[-2], self.layers[-1]) * 0.1

        self.optimizer.set_weights_arrays(self.hidden_weights, self.weights_out)
        
    def forwardPass(self, X: np.ndarray):
        """ Forward pass of MLP """

        self.X = X

        self.hidden_zs = []
        self.activations = [X]
        for i in range(len(self.layers) - 2):
            z = self.activations[-1] @ self.hidden_weights[i]
            a = self.activation.apply(z)

            self.hidden_zs.append(z)
            self.activations.append(a)

        self.z_out = self.activations[-1] @ self.weights_out
        self.y_hat = self.activation.apply(self.z_out)

    def backwardPass(self, y: np.ndarray):
        """ Backward pass of MLP """

        out_error = y - self.y_hat
        out_dz = out_error * self.activation.apply_derivative(self.z_out)
        self.out_dw = self.activations[-1].T @ out_dz

        self.hidden_dws = []
        prev_dz = out_dz
        prev_weights = self.weights_out
        for i in reversed(range(len(self.layers) - 2)):
            error = prev_dz.dot(prev_weights.T)
            dz = error * self.activation.apply_derivative(self.hidden_zs[i])
            dw = self.activations[i].T @ dz

            self.hidden_dws.insert(0, dw)
            prev_dz = dz
            prev_weights = self.hidden_weights[i]

    def update(self):
        """ The update method to update parameters """

        self.optimizer.update(self.hidden_dws, self.out_dw)
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """ Fit method for MLP, call it to train your MLP model """

        for i in range(self.n_epoch):
            sample_indices = np.random.choice(range(0, len(X_train)), size = 30)

            X = np.array([X_train[i] for i in sample_indices])
            y = np.array([y_train[i] for i in sample_indices])

            self.forwardPass(X)
            self.backwardPass(y)
            self.update()

            if i % 300 == 0:
                print(f"Loss: {(y - self.y_hat ** 2).mean()}")
                # for (y, y_hat) in zip(y, self.y_hat):
                #     print(f"\ty: {y}, y_hat: {y_hat}")

    def predict(self, X_test):
        """ Method for predicting class of the testing data """
        y_hat = self.predict_proba(X_test)
        return np.array([1 if i > 0.5 else 0 for i in y_hat])
    
    def predict_proba(self, X_test):
        """ Method for predicting the probability of the testing data """
        self.forwardPass(X_test)
        return self.y_hat


    