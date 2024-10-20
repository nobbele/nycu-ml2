from dataclasses import dataclass
from typing import Any, Callable, Generator, TypeAlias
import numpy as np
from abc import ABC, abstractmethod

# ====== Activation funtion ====== #
class Activation():
    def __init__(self):
        pass
    
    def sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

# ====== Optimizer function ====== #
class Optimizer():
    W: np.ndarray[Any, np.dtype[float]]
    b: float = 0

    lr: float

    def __init__(self):
        pass

    def SGD(self, dW: np.ndarray[Any, np.dtype[float]], db: float = 0):
        self.W = self.W - self.lr * dW
        self.b = self.b - self.lr * db
    


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

@dataclass
class Perceptron:
    w: np.ndarray[Any, np.dtype[float]]
    # bias: float

    z: float = 0
    a: float = 0

@dataclass
class Layer:
    perceptrons: np.ndarray[Any, np.dtype[Perceptron]]

@dataclass
class Network:
    layers: np.ndarray[Any, np.dtype[Perceptron]]

class MLPClassifier(Classifier):
    network: Network
    activate_function: Callable[[float], float]
    optimizer: Optimizer
    n_epoch: int

    def __init__(
        self, 
        network: Network, 
        activate_function: Callable[[float], float], 
        optimizer: Optimizer, 
        learning_rate: float, 
        n_epoch: int = 1000
    ):
        self.network = network
        self.activate_function = activate_function
        self.optimizer = optimizer
        self.optimizer.lr = learning_rate
        self.n_epoch = n_epoch
        
    def forwardPass(
        self, 
        X: np.ndarray[Any, np.ndarray[Any, np.dtype[float]]]
    ) -> Generator[np.ndarray[Any, np.dtype[float]], None, None]:
        """ Forward pass of MLP """

        for sample in X:
            assert isinstance(sample, np.ndarray)

            prev_layer_output = sample
            for layer in self.network.layers:
                assert isinstance(layer, Layer)

                for perceptron in layer.perceptrons:
                    assert isinstance(perceptron, Perceptron)
                    assert len(perceptron.w) == len(prev_layer_output), \
                        "Previous layer's output count didn't match this layer's input count.\n\t" \
                        f"Got input length of {len(prev_layer_output)}, expected {len(perceptron.w)}"
                    
                    perceptron.z = perceptron.w.dot(prev_layer_output)
                    perceptron.a = self.activate_function(perceptron.z)

                prev_layer_output = np.array([perceptron.a for perceptron in layer.perceptrons])
            
            yield prev_layer_output


    def backwardPass(self, y):
        """ Backward pass of MLP """
        # TODO
        pass

    def update(self):
        """ The update method to update parameters """
        # TODO
        pass
    
    def fit(
        self, 
        X_train: np.ndarray[Any, np.ndarray[Any, np.dtype[float]]], 
        y_train: np.ndarray[Any, np.dtype[float]]
    ):
        """ Fit method for MLP, call it to train your MLP model """
        assert len(X_train) == len(y_train)

        # random_sample = np.random.choice([(X_train[i], y_train[i]) for i in range(0, len(X_train))], size=10)
        # X_train = [x for x, y in random_sample]
        # y_train = [y for x, y in random_sample]

        for i in range(0, len(X_train)):
            X = X_train[i]
            assert isinstance(X, np.ndarray)

            y_hat = next(self.forwardPass([X]))
            y = y_train[i]
            assert isinstance(y, np.ndarray)
            assert isinstance(y_hat, np.ndarray)

            loss = y_hat - y
            assert isinstance(loss, np.ndarray)
            print(f"y_hat: {y_hat}, y: {y}")

            dzs = loss
            for i in reversed(range(0, len(self.network.layers))):
                layer = self.network.layers[i]
                assert isinstance(layer, Layer)

                prev_dz = dzs
                dzs = []
                for j in range(0, len(layer.perceptrons)):
                    perceptron = layer.perceptrons[j]
                    assert isinstance(perceptron, Perceptron)

                    input = np.array([perceptron.a for perceptron in self.network.layers[i - 1].perceptrons]) if i >= 1 \
                        else X
                    next_layer_W = np.array([perceptron.w for perceptron in self.network.layers[i + 1].perceptrons]) if i < len(self.network.layers) - 1 \
                        else np.ones((len(y), len(y)))

                    # TODO bias
                    dz = sum([next_layer_W[k][j] * prev_dz[k] for k in range(0, len(next_layer_W))]) * perceptron.a * (1 - perceptron.a)
                    dzs.append(dz)
                    dW = dz * input

                    self.optimizer.W = perceptron.w
                    self.optimizer.SGD(dW)
                    perceptron.w = self.optimizer.W
                    
    def predict(self, X_test):
        """ Method for predicting class of the testing data """
        y_hat = self.predict_proba(X_test)
        return np.array([1 if i > 0.5 else 0 for i in y_hat])
    
    def predict_proba(self, X_test):
        """ Method for predicting the probability of the testing data """
        return self.forwardPass(X_test)


    