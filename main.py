import numpy as np
import pandas as pd
from preprocessor import Preprocessor
from model import Layer, MLPClassifier, Activation, Network, Optimizer, Perceptron
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
import os


def dataPreprocessing():
    """ TODO, use your own dataPreprocess function here. """
     
    # root_path = r"E:\2024_ml\assignment\hw2\proj2_data" # change the root path 
    root_path = r"/home/nobbele/Projects/ML/Project2/pycode/proj2_data/"
    
    train_X = os.path.join(root_path, "train_X.csv")
    train_y = os.path.join(root_path, "train_y.csv")
    test_X = os.path.join(root_path, "test_X.csv")
    test_y = os.path.join(root_path, "test_y.csv")

    train_X = pd.read_csv(train_X)
    train_y = pd.read_csv(train_y)
    test_X = pd.read_csv(test_X)
    test_y = pd.read_csv(test_y)

    def preprocess_X(df):
        pp = Preprocessor(df)
        pp.remove_index()
        pp.fillna()
        pp.yesno_to_int()
        return pp.df.to_numpy()
    
    def preprocess_y(df):
        pp = Preprocessor(df)
        pp.remove_index()
        return pp.df.to_numpy()

    train_X = preprocess_X(train_X)
    train_y = preprocess_y(train_y)

    test_X = preprocess_X(test_X)
    test_y = preprocess_y(test_y)

    return train_X, train_y, test_X, test_y # train, test data should be numpy array


def main():
    train_X, train_y, test_X, test_y = dataPreprocessing() # train, test data should not contain index

    input_layer_N = 77
    hidden_layer1_N = 50
    hidden_layer2_N = 25
    output_layer_N = 1

    model = MLPClassifier(
        Network(np.array([
            Layer([Perceptron(np.random.randn(input_layer_N)) for _ in range(0, hidden_layer1_N)]),
            Layer([Perceptron(np.random.randn(hidden_layer1_N)) for _ in range(0, hidden_layer2_N)]),
            Layer([Perceptron(np.random.randn(hidden_layer2_N)) for _ in range(0, output_layer_N)]),
        ])), 
        Activation().sigmoid, 
        Optimizer(), 
        0.01
    )
    model.fit(train_X, train_y)
    pred = model.predict(test_X)

    acc = accuracy_score(pred, test_y)
    f1 = f1_score(pred, test_y, zero_division=0)
    mcc = matthews_corrcoef(pred, test_y)

    print(f'Acc: {acc:.5f}')
    print(f'F1 score: {f1:.5f}')
    print(f'MCC: {mcc:.5f}')
    scoring = 0.3 * acc + 0.35 * f1 + 0.35 * mcc
    print(f'Scoring: {scoring:.5f}')


if __name__ == "__main__":
    np.random.seed(0)
    main()
    

