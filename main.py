import numpy as np
import pandas as pd
from preprocessor import Preprocessor
from model import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
import os


def dataPreprocessing():     
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
        pp.standardize()
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

    # train_X = np.array([
    #     np.array([0, 0, 0, 0]),
    #     np.array([1, 0, 0, 0]),
    #     np.array([0, 1, 0, 0]),
    #     np.array([0, 1, 1, 0]),
    #     np.array([1, 0, 0, 1]),
    # ])
    # train_y = np.array([
    #     np.array([0]),
    #     np.array([0]),
    #     np.array([0]),
    #     np.array([1]),
    #     np.array([1]),
    # ])

    # test_X = np.array([
    #     np.array([0, 0, 1, 1]),
    #     np.array([1, 0, 0, 1]),
    # ])
    # test_y = np.array([
    #     np.array([0]),
    #     np.array([1]),
    # ])

    train_X = np.array([
        np.array([0, 0]),
        np.array([0, 1]),
        np.array([1, 0]),
        np.array([1, 1]),
    ])
    train_y = np.array([
        np.array([0]),
        np.array([0]),
        np.array([0]),
        np.array([1]),
    ])
    # test_X = np.array([
    #     np.array([400]),
    #     np.array([600]),
    # ])
    # test_y = np.array([
    #     np.array([20_000]),
    #     np.array([30_000]),
    # ])

    model = MLPClassifier(
        # [77, 50, 50, 1], 
        [2, 0, 4, 1], 
        0.01
    )
    model.fit(train_X, train_y)
    # print(model.forwardPass([
    #     train_X[0],
    # ]))
    # model.fit(np.array([
    #     train_X[10],
    #     train_X[6],
    #     train_X[8],
    #     train_X[3],
    #     train_X[5],
    # ]), np.array([
    #     train_y[10],
    #     train_y[6],
    #     train_y[8],
    #     train_y[3],
    #     train_y[5],
    # ]))
    # pred = model.predict(test_X)

    # acc = accuracy_score(pred, test_y)
    # f1 = f1_score(pred, test_y, zero_division=0)
    # mcc = matthews_corrcoef(pred, test_y)

    # print(f'Acc: {acc:.5f}')
    # print(f'F1 score: {f1:.5f}')
    # print(f'MCC: {mcc:.5f}')
    # scoring = 0.3 * acc + 0.35 * f1 + 0.35 * mcc
    # print(f'Scoring: {scoring:.5f}')


if __name__ == "__main__":
    np.random.seed(5)
    main()
    

