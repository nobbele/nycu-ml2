from typing import Any, Callable
import numpy as np
from abc import ABC, abstractmethod

# ====== Activation funtion ====== #
class Activation():
    def __init__(self):
        pass
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))
    
    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
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
    
def meanSquareError(pred: np.ndarray, target: np.ndarray):
    return np.sum((pred - target) ** 2) / 2

def meanSquareErrorDerivative(pred: np.ndarray, target: np.ndarray):
    return pred - target

# def binaryCrossEntropy(pred: np.ndarray, target: np.ndarray):
#     return -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))


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

    # [ <Node 0 Weights>
    #   <Node 1 Weights>
    #   <Node 2 Weights>, ... ]
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

        self.weights = [np.random.rand(self.layers[i], self.layers[i - 1]) / 10 for i in range(1, len(self.layers))]
        self.biases = [np.zeros((1, self.layers[i + 1])) for i in range(len(self.layers) - 1)]

        
    def forwardPass(
        self, 
        X: np.ndarray
    ) -> (list[tuple[np.ndarray, np.ndarray]]):
        """ Forward pass of MLP """

        # 10 rows    of samples
        # 77  columns of features
        # assert X.shape == (10, 77)

        sample_count = X.shape[0]

        activations = [X]
        for W, b in zip(self.weights, self.biases):
            # [ <Sample 0 Activations>
            #   <Sample 1 Activations>
            #   <Sample 2 Activations>, ... ]
            I = activations[-1]

            # [ <Z_0, Z_1, Z_2, ...> Sample 0 Zs for Node 0, 1, 2, ...
            #   <Z_0, Z_1, Z_2, ...> Sample 1 Zs for Node 0, 1, 2, ...
            #   <Z_0, Z_1, Z_2, ...>, ... ]
            z = I @ W.T + b
            assert z.shape == (sample_count, len(W))

            # print(f"z = {z}")

            # [ <A_0, A_1, A_2, ...> Sample 0 Activations for Node 0, 1, 2, ...
            #   <A_0, A_1, A_2, ...> Sample 1 Activations for Node 0, 1, 2, ...
            #   <A_0, A_1, A_2, ...>, ... ]
            A = self.activation.sigmoid(z)
            assert A.shape == (sample_count, len(W))

            activations.append(A)
        return activations


    def backwardPass(
            self, 
            X: np.ndarray[np.ndarray], 
            y: np.ndarray[np.ndarray], 
            activations: list[np.ndarray]
        ) -> list[tuple[np.ndarray, np.ndarray]]:
        """ Backward pass of MLP """

        sample_count = len(y)
        y_hat = activations[-1]

        # Output layer to last hidden layer
        error = meanSquareErrorDerivative(y_hat, y)

        dZ = error * y_hat * (1 - y_hat)

        # for (ny, ny_hat, ne, ndz) in zip(y, y_hat, error, dZ):
        #     print(f"y: {ny} - y_hat: {ny_hat} = error: {ne}, dZ = {ndz}")

        # print()

        # activations[-1]: Output Activation (y_hat)
        # activations[-2]: Hidden Layer 2 Activations
        # activations[-3]: Hidden Layer 1 Activations
        # activations[-4]: Input Layer Activations
        
        # Calculate the gradients for this layer
        dW = ((activations[-2].T @ dZ) / sample_count).T

        # print(f"activations[-2]: {activations[-2]}")
        # print(f"dW: {dW}")

        # print(f"dW: {dW}")

        db = dZ.sum(axis=0) / sample_count
        # db = 0

        gradients = [(dW, db)]

        # Hidden layers, start at 1 cause input doesn't have weights
        for l in reversed(range(0, len(activations) - 2)):
            error = dZ @ self.weights[l + 1]

            # same dz calculation for current layer
            dZ = error * activations[l + 1] * (1 - activations[l + 1])

            # for (ny, ny_hat, ne, ndz) in zip(y, y_hat, error, dZ):
            #     print(f"y: {ny} - y_hat: {ny_hat} = error: {ne}, dZ = {ndz}")

            # Calculate the gradients for this layer, same as before
            # activations[l] is the previous layer's activation (since weight[0] is the weights between layer 0 and 1).
            dW = (activations[l].T @ dZ).T / sample_count
            db = dZ.sum(axis=0) / sample_count

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

        # print(f"weights: {self.weights}")
        print()

        for epoch in range(self.n_epoch):
        # for epoch in range(4):
            sample_indices = np.random.choice(range(0, len(X_train)), size = 10)
            # sample_indices = [i for i in range(10)]
            X = np.array([X_train[i] for i in sample_indices])
            y = np.array([y_train[i] for i in sample_indices])

            # print(f"X: {X}")
            # print(f"y: {y}")

            activations = self.forwardPass(X)
            # print(f"activations: {activations}")
            loss = meanSquareError(activations[-1], y)
            gradients = self.backwardPass(X, y, activations)
            # print(f"before weights: {self.weights}")
            # print(f"gradients: {gradients}")
            self.update(gradients)
            # print(f"after weights: {self.weights}")

            if epoch % 300 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
                # print(f"activations: {activations}")
                for (ny, ny_hat) in zip(y, activations[-1]):
                    print(f"y: {ny} - y_hat: {ny_hat}")
                print()

    def predict(self, X_test):
        """ Method for predicting class of the testing data """
        y_hat = self.predict_proba(X_test)
        return np.array([1 if i > 0.5 else 0 for i in y_hat])
    
    def predict_proba(self, X_test):
        """ Method for predicting the probability of the testing data """
        # Gets the last output, get the A list from the z,A tuple
        return self.forwardPass(X_test)[-1]


    