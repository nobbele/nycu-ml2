from typing import Any, Callable
import numpy as np
from abc import ABC, abstractmethod

# ====== Activation funtion ====== #
class Activation():
    def __init__(self):
        pass
    
    def sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))
    
    def sigmoid_derivative(self, x: float) -> float:
        return self.sigmoid(x) * (1 - self.sigmoid(x))

# ====== Optimizer function ====== #
class Optimizer():
    W: np.ndarray
    b: np.ndarray

    lr: float

    def __init__(self):
        pass

    def SGD(self, dW: np.ndarray, db: np.ndarray):
        self.W -= self.lr * dW
        self.b -= self.lr * db
    
def binaryCrossEntropy(pred: np.ndarray, target: np.ndarray):
    return -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))


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
    layers: list[int]
    activation: Activation
    optimizer: Optimizer
    n_epoch: int

    weights: list[np.ndarray]
    biases: list[np.ndarray]

    def __init__(
        self, 
        layers: list[int], 
        activation: Activation, 
        optimizer: Optimizer, 
        learning_rate: float, 
        n_epoch: int = 1000
    ):
        self.layers = layers
        self.activation = activation
        self.optimizer = optimizer
        self.optimizer.lr = learning_rate
        self.n_epoch = n_epoch

        self.weights = [np.random.rand(self.layers[i], self.layers[i + 1]) for i in range(len(self.layers) - 1)]
        self.biases = [np.random.rand(1, self.layers[i + 1]) for i in range(len(self.layers) - 1)]

        
    def forwardPass(
        self, 
        X: np.ndarray
    ) -> (list[tuple[np.ndarray, np.ndarray]]):
        """ Forward pass of MLP """

        activations = [(0, X)]
        for W, b in zip(self.weights, self.biases):
            z = activations[-1][1].dot(W) + b
            A = self.activation.sigmoid(z)
            activations.append((z, A))
        return activations


    def backwardPass(
            self, 
            X: np.ndarray[np.ndarray], 
            y: np.ndarray[np.ndarray], 
            activations: list[np.ndarray]
        ) -> list[tuple[np.ndarray, np.ndarray]]:
        """ Backward pass of MLP """

        sample_count = len(y)

        # Output layer to last hidden layer
        error = activations[-1][1] - y
        # dL/dy_hat * dy_hat/dz
        # dL/dz
        dZ = error * self.activation.sigmoid_derivative(activations[-1][0])

        # activations[-1]: Output Activation
        # activations[-2]: Hidden Layer 2 Activations
        # activations[-3]: Hidden Layer 1 Activations
        # activations[-4]: Input Layer Activations
        
        # Calculate the gradients for this layer
        dW = activations[-2][1].T.dot(dZ) / sample_count
        db = dZ.sum(axis=0) / sample_count
        # dW = dL/z^(out) = dL/dy_hat * dy_hat/dz^(out) = error * sigmoid'(z^(out))
        gradients = [(dW, db)]

        # Hidden layers, start at 2 cause output doesn't have weights and output to last hidden layer was calculated above.
        for l in range(2, len(activations)):
            # print(f"next_weights: {self.weights[-l + 1]}")
            # error = next layer dZ * next layer weights
            error = dZ.dot(self.weights[-l + 1].T)
            # same dz calculation for current layer
            dZ = error * self.activation.sigmoid_derivative(activations[-l][0])

            # print(f"dZ: {dZ}")
            # print(f"error: {error}")
            # print(f"prev_activations: {activations[-l - 1]}")
            # print(f"activations: {activations[-l]}")
            # print(f"next_activations: {activations[-l + 1]}")
            
            # Calculate the gradients for this layer, same as before
            dW = activations[-l - 1][1].T.dot(dZ) / sample_count
            db = dZ.sum(axis=0) / sample_count

            # print("activations[-l - 1][1]")
            # print(activations[-l - 1][1])
            # print(dZ)

            # print(f"dW: {dW}")

            # print(f"dW: {dW}")

            # print(f"Le: {len(activations[-l])}")
            # print(f"L: {len(activations) - l}")
            # print("A")
            # print(activations[-l - 1])
            # print("dW")
            # print(dW)

            gradients.insert(0, (dW, db))

        return gradients

    def update(self, gradients: list[tuple[np.ndarray, np.ndarray]]):
        """ The update method to update parameters """
        for i in range(len(self.weights)):
            self.optimizer.W = self.weights[i]
            self.optimizer.b = self.biases[i]
            self.optimizer.SGD(gradients[i][0], gradients[i][1])
            self.weights[i] = self.optimizer.W
            self.biases[i] = self.optimizer.b
    
    def fit(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray
    ):
        """ Fit method for MLP, call it to train your MLP model """
        assert len(X_train) == len(y_train)

        print(f"weights: {self.weights}")
        print()

        for epoch in range(self.n_epoch):
        # for epoch in range(4):
            sample_indices = np.random.choice(range(0, len(X_train)), size = 10)
            X = np.array([X_train[i] for i in sample_indices])
            y = np.array([y_train[i] for i in sample_indices])

            # print(f"X: {X}")
            # print(f"y: {y}")

            activations = self.forwardPass(X)
            # print(f"activations: {activations}")
            loss = binaryCrossEntropy(activations[-1][1], y)
            gradients = self.backwardPass(X, y, activations)
            # print(f"before weights: {self.weights}")
            # print(f"gradients: {gradients}")
            self.update(gradients)
            # print(f"after weights: {self.weights}")

            if epoch % 300 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
                print(f"y: {np.array(list(zip(y, activations[-1][1])))}")
                print()

    def predict(self, X_test):
        """ Method for predicting class of the testing data """
        y_hat = self.predict_proba(X_test)
        return np.array([1 if i > 0.5 else 0 for i in y_hat])
    
    def predict_proba(self, X_test):
        """ Method for predicting the probability of the testing data """
        # Gets the last output, get the A list from the z,A tuple
        return self.forwardPass(X_test)[-1][1]


    