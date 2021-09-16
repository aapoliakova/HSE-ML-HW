import scipy.linalg
from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
from sklearn.preprocessing import normalize
# X_normalized = normalize(X, norm='l2')

def read_data(path="boston.csv", normilaze=True):
    dataframe = np.genfromtxt(path, delimiter=",", skip_header=15)
    if normilaze:
        dataframe = normalize(dataframe, norm='l2')
    np.random.seed(42)
    np.random.shuffle(dataframe)
    X = dataframe[:, :-1]
    y = dataframe[:, -1]

    return X, y

X, y = read_data()


class GradientLR:
    def __init__(self, alpha: float, iterations=10000, l=0.):
        self.alpha = alpha
        self.iterations = iterations
        self.l = l
        self.w = None
        self.tol = 1e5

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.column_stack((np.ones_like(X[:, 0]), X))
        self.w = np.zeros(X.shape[1])
        print(X.shape, self.w.shape)

        for n in range(self.iterations):
            without_prob = self.l * np.sign(self.w)
            grad = (1 / y.shape[0]) * X.T @ (X @ self.w - y) + without_prob
            self.w = self.w - self.alpha * grad
            if self.w.T @ self.w <= self.tol:
                print(n)
                break

    def predict(self, X: np.ndarray):
        X = np.column_stack((np.ones_like(X[:, 0]), X))
        print(X.shape, self.w.shape)
        return X @ self.w.T

test = GradientLR(alpha=0.0001, l=0.01)
test.fit(X, y)
y_pred = test.predict(X)
print(y_pred)
