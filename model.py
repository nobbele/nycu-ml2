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
    h2_w: np.ndarray
    h1_w: np.ndarray

    def set_weights_arrays(self, h1_w, h2_w, out_w):
        self.h1_w = h1_w
        self.h2_w = h2_w
        self.out_w = out_w

    @abstractmethod
    def update(self, h1_dW, h2_dW, out_dW):
        pass

class SGDOptimizer(Optimizer):
    lr: float

    def __init__(self, lr: float):
        super(Optimizer, self).__init__()
        self.lr = lr

    def update(self, h1_dW, h2_dW, out_dW):
        self.h1_w += h1_dW * self.lr
        self.h2_w += h2_dW * self.lr
        self.out_w += out_dW * self.lr

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

    def __init__(self, layers, activation, optimizer, n_epoch = 10_000):
        """ TODO, Initialize your own MLP class """

        self.layers = layers
        self.activation = activation
        self.optimizer = optimizer
        self.n_epoch = n_epoch

        self.weights_h1 = np.random.rand(self.layers[0], self.layers[1]) * 0.1
        self.weights_h2 = np.random.rand(self.layers[1], self.layers[2]) * 0.1
        self.weights_out = np.random.rand(self.layers[2], self.layers[3]) * 0.1

        self.optimizer.set_weights_arrays(self.weights_h1, self.weights_h2, self.weights_out)
        
    def forwardPass(self, X):
        """ Forward pass of MLP """

        self.X = X

        self.z_h1 = np.dot(self.X, self.weights_h1)
        self.a_h1 = self.activation.apply(self.z_h1)

        self.z_h2 = np.dot(self.a_h1, self.weights_h2)
        self.a_h2 = self.activation.apply(self.z_h2)

        self.z_out = np.dot(self.a_h2, self.weights_out)
        self.y_hat = self.activation.apply(self.z_out)

    def backwardPass(self, y):
        """ Backward pass of MLP """

        out_error = y - self.y_hat
        out_dz = out_error * self.activation.apply_derivative(self.z_out)

        h2_error = out_dz.dot(self.weights_out.T)
        h2_dz = h2_error * self.activation.apply_derivative(self.z_h2)

        h1_error = h2_dz.dot(self.weights_h2.T)
        h1_dz = h1_error * self.activation.apply_derivative(self.z_h1)

        self.out_dw = self.a_h2.T @ out_dz
        self.h2_dw = self.a_h1.T @ h2_dz
        self.h1_dw = self.X.T @ h1_dz

    def update(self):
        """ The update method to update parameters """

        self.optimizer.update(self.h1_dw, self.h2_dw, self.out_dw)
    
    def fit(self, X_train, y_train):
        """ Fit method for MLP, call it to train your MLP model """

        for i in range(self.n_epoch):
            sample_indices = np.random.choice(range(0, len(X_train)), size = 10)

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


    