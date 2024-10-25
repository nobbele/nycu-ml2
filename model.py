import numpy as np
from abc import ABC, abstractmethod

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def meanSquareError(pred: np.ndarray, target: np.ndarray):
    n = len(pred)
    return np.sum((pred - target) ** 2) / (2 * n)

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
    n_epoch: int

    weights_h1: np.ndarray
    prev_dW_h1: np.ndarray
    # weights_h2: np.ndarray
    weights_out: np.ndarray
    prev_dW_out: np.ndarray

    lr: float

    def __init__(
        self, 
        layers: list[int], 
        learning_rate: float, 
        n_epoch: int = 100
    ):
        self.layers = layers
        self.lr = learning_rate
        self.n_epoch = n_epoch

        # self.weights_h1 = np.random.rand(self.layers[0], self.layers[1])
        self.weights_h1 = np.random.rand(self.layers[0], self.layers[2])
        self.prev_dW_h1 = np.zeros((self.layers[0],  self.layers[2]))
        # self.weights_h2 = np.random.rand(self.layers[1], self.layers[2])
        self.weights_out = np.random.rand(self.layers[2], self.layers[3])
        self.prev_dW_out = np.zeros((self.layers[2],  self.layers[3]))

        
    def forwardPass(
        self, 
        X: np.ndarray
    ) -> np.ndarray:
        """ Forward pass of MLP """

        predictions = np.array([])
        for x in X:
            A_h1 = sigmoid(x @ self.weights_h1)
            # A_h2 = sigmoid(A_h2 @ self.weights_h2)
            y_hat = sigmoid(A_h1 @ self.weights_out)
            predictions = np.append(predictions, y_hat)

        return predictions

    def backwardPass(
            self, 
            Xs: np.ndarray[np.ndarray], 
            ys: np.ndarray[float], 
        ) -> list[tuple[np.ndarray, np.ndarray]]:
        """ Backward pass of MLP """

        sample_changes = []

        for (x, y) in zip(Xs, ys):
            z_h1 = x @ self.weights_h1
            A_h1 = sigmoid(z_h1)
            # A_h2 = sigmoid(A_h2 @ self.weights_h2)
            z_out = A_h1 @ self.weights_out
            y_hat = sigmoid(z_out)

            # Calculate gradient for output layer
            dW_out = np.array([])
            for weight_i in range(self.weights_out.shape[0]):
                delta = (y_hat - y) * (sigmoid(z_out[0]) * (1 - sigmoid(z_out[0]))) * A_h1[weight_i]
                dW_out = np.append(dW_out, delta)

            # Calculate gradient for hidden layer
            dWs_h1 = []
            for node_i in range(self.weights_h1.shape[1]):
                dW = np.array([])
                for weight_i in range(self.weights_h1.shape[0]):
                    delta_a = (y_hat - y) * (sigmoid(z_out[0]) * (1 - sigmoid(z_out[0]))) * self.weights_out[node_i]
                    delta_z = delta_a * (sigmoid(z_h1[node_i]) * (1 - sigmoid(z_h1[node_i])))
                    delta = delta_z.mean() * x[weight_i]
                    dW = np.append(dW, delta)
                dWs_h1.append(dW)

            sample_changes.append((dWs_h1, dW_out))

        return sample_changes
    


    def update(self, gradients: list[tuple[np.ndarray, np.ndarray]]):
        """ The update method to update parameters """

        sample_count = len(gradients)

        alpha = 1
        momentum = 1

        for (gradients_h1, gradients_out) in gradients:
            for node_i in range(len(gradients_h1)):
                for weight_i in range(len(gradients_h1)):
                    delta = alpha * gradients_h1[node_i][weight_i] / sample_count
                    self.weights_h1[weight_i][node_i] -= delta
                    self.prev_dW_h1[weight_i][node_i] = delta

            for weight_i in range(len(self.weights_out)):
                delta = alpha * gradients_out[weight_i]  / sample_count
                self.weights_out[weight_i] -= delta
                self.prev_dW_out[weight_i] = delta

    def fit(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray
    ):
        """ Fit method for MLP, call it to train your MLP model """
        y_hats = self.forwardPass(X_train)
        error_before = meanSquareError(y_hats, y_train)
        print(f"Error before: {error_before}")

        for epoch in range(self.n_epoch):
            sample_indices = np.random.choice(range(0, len(X_train)), size = 10)
            # sample_indices = [i for i in range(10)]
            X = np.array([X_train[i] for i in sample_indices])
            y = np.array([float(y_train[i]) for i in sample_indices])

            gradients = self.backwardPass(X, y)
            self.update(gradients)

            if epoch % 30 == 0:
                y_hats = self.forwardPass(X_train)
                loss = meanSquareError(y_hats, y_train)
                print(f"Epoch {epoch}, Loss: {loss}")

        sample_indices = np.random.choice(range(0, len(X_train)), size = 10)
        # sample_indices = [i for i in range(10)]
        X = np.array([X_train[i] for i in sample_indices])
        y = np.array([float(y_train[i]) for i in sample_indices])

        y_hats = self.forwardPass(X)
        print(y_hats)
        print(y)
        error_after = meanSquareError(y_hats, y)
        print(f"Error after: {error_after}")
        print(f"difference: {error_after - error_before}")

    def predict(self, X_test):
        """ Method for predicting class of the testing data """
        y_hat = self.predict_proba(X_test)
        return np.array([1 if i > 0.5 else 0 for i in y_hat])
    
    def predict_proba(self, X_test):
        """ Method for predicting the probability of the testing data """
        return self.forwardPass(X_test)


    