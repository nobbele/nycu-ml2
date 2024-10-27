import numpy as np
import pandas as pd
from preprocessor import Preprocessor
from model import AdaGradOptimizer, IdentityActivation, LeakyReLUActivation, MLPClassifier, MomentumOptimizer, SGDOptimizer, SigmoidActivation
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
        pp.feature_selection()
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

    # [77, 50, 50, 1], 
    # LeakyReLUActivation(0.1),
    # SGDOptimizer(lr = 0.0007)
    # Result: 0.60569

    # [77, 50, 50, 1], 
    # IdentityActivation(),
    # SGDOptimizer(lr = 0.0007)
    # Result: 0.69210

    model = MLPClassifier(
        [74, 55, 55, 1], 
        # IdentityActivation(),
        SigmoidActivation(),
        # LeakyReLUActivation(0.2),
        # SGDOptimizer(lr = 0.00015),
        # MomentumOptimizer(lr = 0.0002, beta = 0.9),
        AdaGradOptimizer(lr = 0.016),
        10_000,
    )
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    # print(accuracy_score(model.predict(train_X), train_y))

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
    

