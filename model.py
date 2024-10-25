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
    
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    return sigmoid(x) * (1 - sigmoid(x))

# ====== Optimizer function ====== #
class Optimizer():
    W: np.ndarray
    b: np.ndarray

    lr: float

    def __init__(self):
        pass

    def SGD(self, dW: np.ndarray, db: np.ndarray = None):
        self.W -= self.lr * dW
        if db != None:
            self.b -= self.lr * db
    
def meanSquareError(pred: np.ndarray, target: np.ndarray):
    n = len(pred)
    return np.sum((pred - target) ** 2) / n

print(meanSquareError(np.array([1, 2, 3, 4]), np.array([0, 1, 2, 2.9])))

def meanSquareErrorDerivative(pred: np.ndarray, target: np.ndarray):
    n = len(pred)
    return 2 * (pred - target) / n

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

    weights_h1: np.ndarray
    weights_h2: np.ndarray
    weights_out: np.ndarray

    activation_h1: np.ndarray
    activation_h2: np.ndarray
    y_hat: np.ndarray

    # biases: list[np.ndarray]

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

        self.weights_h1 = np.random.rand(self.layers[1], self.layers[0]) / 20
        self.weights_h2 = np.random.rand(self.layers[2], self.layers[1]) / 20
        self.weights_out = np.random.rand(self.layers[3], self.layers[2]) / 20

        # self.biases = [np.zeros((1, self.layers[i + 1])) for i in range(len(self.layers) - 1)]

        
    def forwardPass(
        self, 
        X: np.ndarray
    ):
        """ Forward pass of MLP """

        # print(X.shape)
        # print(self.weights_h1.shape)

        self.activation_h1 = sigmoid(X @ self.weights_h1.T)
        self.activation_h2 = sigmoid(self.activation_h1 @ self.weights_h2.T)
        self.y_hat = sigmoid(self.activation_h2 @ self.weights_out.T)

    def backwardPass(
            self, 
            X: np.ndarray[np.ndarray], 
            y: np.ndarray[float], 
        ) -> list[tuple[np.ndarray, np.ndarray]]:
        """ Backward pass of MLP """

        # sample_count = len(y)
        print(self.y_hat)
        print(y)
        print()

        error_out = meanSquareErrorDerivative(self.y_hat, y)
        dZ_out = error_out * (self.y_hat * (1 - self.y_hat))
        dZ_out = np.array([np.array([n]) for n in dZ_out])
        dW_out = self.activation_h2.T @ dZ_out / len(self.y_hat)
        assert dW_out.shape == (self.layers[2],1)
        # print(f"dZ_out: {dZ_out.shape}")
        # print(f"self.activation_h2: {self.activation_h2.shape}")
        # print(f"dW_out: {dW_out.shape}")

        # print(f"self.weights_out: {self.weights_out}")
        # print(f"dZ_out: {dZ_out}")

        dW_h2 = []
        dZ_h2 = 0
        for i in range(self.layers[2]):
            error = dZ_out @ self.weights_out.T[i]
            dZ = error * (self.activation_h2.T[i] * (1 - self.activation_h2.T[i]))
            print(dZ.shape)
            dZ_h2 += dZ
            dW = float((self.activation_h1.T @ dZ).mean())
            dW_h2.append(dW)
        dW_h2 = np.array(dW_h2)
        dZ_h2 /= self.layers[2]

        dW_h1 = []
        dZ_h1 = 0
        for i in range(self.layers[2]):
            error = dZ_h2 @ self.weights_h2.T[i]
            dZ = error * (self.activation_h1.T[i] * (1 - self.activation_h1.T[i]))
            dZ_h1 += dZ
            dW = float((X.T @ dZ).mean())
            dW_h1.append(dW)
        dW_h1 = np.array(dW_h1)
        dZ_h1 /= self.layers[2]

        # print(self.weights_out.shape)
        # print(dZ_out.shape)
        # error_h2 = self.weights_out * dZ_out.mean()
        # print(error_h2.shape)
        # # print(f"self.weights_out: {self.weights_out.shape}")
        # # print(f"error_h2.shape: {error_h2.shape}")
        # # print(f"error_h2: {error_h2}")
        # dZ_h2 = error_h2 * self.activation_h2 * (1 - self.activation_h2)
        # print(f"dZ_h2: {dZ_h2.shape}")
        # dW_h2 = self.activation_h1.T @ dZ_h2 / len(self.activation_h2)
        # print(f"dW_h2.shape: {dW_h2.shape}")
        # assert dW_h2.shape == (self.layers[1],)
        # # print(f"dW_h2: {dW_h2}")
        # print()

        # print(self.weights_h2.shape)
        # print(dZ_h2.shape)
        # error_h1 =  self.weights_h2.T @ dZ_h2
        # print(error_h2.shape)

        gradients = [dW_h1, dW_h2, dW_out]

        

        return gradients

        # sample_count = len(y)
        # y_hat = activations[-1]

        # Output layer to last hidden layer 
        # error = meanSquareErrorDerivative(y_hat, y)

        # dZ = error * y_hat * (1 - y_hat)

        # for (ny, ny_hat, ne, ndz) in zip(y, y_hat, error, dZ):
        #     print(f"y: {ny} - y_hat: {ny_hat} = error: {ne}, dZ = {ndz}")

        # print()

        # activations[-1]: Output Activation (y_hat)
        # activations[-2]: Hidden Layer 2 Activations
        # activations[-3]: Hidden Layer 1 Activations
        # activations[-4]: Input Layer Activations
        
        # Calculate the gradients for this layer
        # dW = (activations[-2].T @ dZ) / sample_count

        # # print(f"activations[-2]: {activations[-2]}")
        # # print(f"dW: {dW}")

        # # print(f"dW: {dW}")

        # db = dZ.sum(axis=0) / sample_count
        # # db = 0

        # gradients = [(dW, db)]

        # for l in reversed(range(len(self.weights) - 1)):
        #     error = dZ @ self.weights[l + 1].T

        #     # same dz calculation for current layer
        #     dZ = error * activations[l + 1] * (1 - activations[l + 1])

        #     # for (ny, ny_hat, ne, ndz) in zip(y, y_hat, error, dZ):
        #     #     print(f"y: {ny} - y_hat: {ny_hat} = error: {ne}, dZ = {ndz}")

        #     # Calculate the gradients for this layer, same as before
        #     # activations[l] is the previous layer's activation (since weight[0] is the weights between layer 0 and 1).
        #     dW = (activations[l].T @ dZ) / sample_count
        #     db = dZ.sum(axis=0) / sample_count

        #     gradients.insert(0, (dW, db))

        # assert len(gradients) == len(self.weights)
        # return gradients

    def update(self, gradients: list[tuple[np.ndarray, np.ndarray]]):
        """ The update method to update parameters """

        # print(self.weights_h1.shape)
        # print(gradients[0].shape)

        self.optimizer.W = self.weights_h1
        self.optimizer.SGD(gradients[0])
        self.weights_h1 = self.optimizer.W

        self.optimizer.W = self.weights_h2
        self.optimizer.SGD(gradients[1])
        self.weights_h2 = self.optimizer.W

        self.optimizer.W = self.weights_out
        self.optimizer.SGD(gradients[2])
        self.weights_out = self.optimizer.W
    
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
            y = np.array([float(y_train[i]) for i in sample_indices])

            # print(f"X: {X}")
            # print(f"y: {y}")

            self.forwardPass(X)
            # print(f"activations: {activations}")
            loss = meanSquareError(self.y_hat, y)
            gradients = self.backwardPass(X, y)
            # print(f"before weights: {self.weights}")
            # print(f"gradients: {gradients}")
            self.update(gradients)
            # print(f"after weights: {self.weights[0].min()} - {self.weights[0].max()}")
            # print(f"after biases: {self.biases}")

            if epoch % 300 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
                # print(f"activations: {activations}")
                for (ny, ny_hat) in zip(y, self.y_hat):
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


    