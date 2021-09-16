import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import datasets
import copy
from typing import NoReturn


def visualize(X, labels_true, labels_pred, w):
    unique_labels = np.unique(labels_true)
    unique_colors = dict([(l, c) for l, c in zip(unique_labels, [[0.8, 0., 0.], [0., 0., 0.8]])])
    plt.figure(figsize=(9, 9))

    if w[1] == 0:
        plt.plot([X[:, 0].min(), X[:, 0].max()], w[0] / w[2])
    elif w[2] == 0:
        plt.plot(w[0] / w[1], [X[:, 1].min(), X[:, 1].max()])
    else:
        mins, maxs = X.min(axis=0), X.max(axis=0)
        pts = [[mins[0], -mins[0] * w[1] / w[2] - w[0] / w[2]],
               [maxs[0], -maxs[0] * w[1] / w[2] - w[0] / w[2]],
               [-mins[1] * w[2] / w[1] - w[0] / w[1], mins[1]],
               [-maxs[1] * w[2] / w[1] - w[0] / w[1], maxs[1]]]
        pts = [(x, y) for x, y in pts if mins[0] <= x <= maxs[0] and mins[1] <= y <= maxs[1]]
        x, y = list(zip(*pts))
        plt.plot(x, y, c=(0.75, 0.75, 0.75), linestyle="--")

    colors_inner = [unique_colors[l] for l in labels_true]
    colors_outer = [unique_colors[l] for l in labels_pred]
    plt.scatter(X[:, 0], X[:, 1], c=colors_inner, edgecolors=colors_outer)
    plt.show()


class Perceptron:
    def __init__(self, iterations: int = 100):
        """
        Parameters
        ----------
        iterations : int
        Количество итераций обучения перцептрона.

        Attributes
        ----------
        w : np.ndarray
        Веса перцептрона размерности X.shape[1] + 1 (X --- данные для обучения),
        w[0] должен соответстовать константе,
        w[1:] - коэффициентам компонент элемента X.

        Notes
        -----
        Вы можете добавлять свои поля в класс.

        """

        self.iterations = iterations
        self.w = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает простой перцептрон.
        Для этого сначала инициализирует веса перцептрона,
        а затем обновляет их в течении iterations итераций.

        Parameters
        ----------
        X : np.ndarray
            Набор данных, на котором обучается перцептрон.
        y: np.ndarray
            Набор меток классов для данных.

        """
        n_samples, n_features = X.shape
        self.w = np.zeros((n_features + 1))
        X = np.column_stack((np.ones_like(X[:, 0]), X))

        self.label_1, self.label_2 = np.unique(y)
        self.y = np.copy(y)
        self.y = np.select([self.y == self.label_1, self.y == self.label_2], [-1, 1])

        for _ in range(self.iterations):

            y_pred = np.sign(X @ self.w.T)

            for i in range(n_samples):
                if y_pred[i] != self.y[i]:
                    self.w = self.w + self.y[i] * X[i]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки классов.

        Parameters
        ----------
        X : np.ndarray
            Набор данных, для которого необходимо вернуть метки классов.

        Return
        ------
        labels : np.ndarray
            Вектор индексов классов
            (по одной метке для каждого элемента из X).

        """
        X = np.column_stack((np.ones_like(X[:, 0]), X))
        y_pred = np.sign(X @ self.w.T).astype(int)
        y_pred = np.select([y_pred == -1, y_pred == 1], [self.label_1, self.label_2])

        return y_pred


class PerceptronBest:

    def __init__(self, iterations: int = 100):
        """
        Parameters
        ----------
        iterations : int
        Количество итераций обучения перцептрона.

        Attributes
        ----------
        w : np.ndarray
        Веса перцептрона размерности X.shape[1] + 1 (X --- данные для обучения),
        w[0] должен соответстовать константе,
        w[1:] - коэффициентам компонент элемента X.

        Notes
        -----
        Вы можете добавлять свои поля в класс.

        """

        self.w = None
        self.iterations = iterations

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает перцептрон.

        Для этого сначала инициализирует веса перцептрона,
        а затем обновляет их в течении iterations итераций.

        При этом в конце обучения оставляет веса,
        при которых значение accuracy было наибольшим.

        Parameters
        ----------
        X : np.ndarray
            Набор данных, на котором обучается перцептрон.
        y: np.ndarray
            Набор меток классов для данных.

        """
        n_samples, n_features = X.shape
        self.w = np.zeros((n_features + 1))
        X = np.column_stack((np.ones_like(X[:, 0]), X))

        self.label_1, self.label_2 = np.unique(y)
        self.y = np.copy(y)
        self.y = np.select([self.y == self.label_1, self.y == self.label_2], [-1, 1])

        scores = []
        weights = []

        for _ in range(self.iterations):

            y_pred = np.sign(X @ self.w.T)
            accuracy = np.mean(y_pred == self.y)
            scores.append(accuracy)
            weights.append(self.w)

            for i in range(n_samples):
                if y_pred[i] != y[i]:
                    self.w = self.w + self.y[i] * X[i]

        self.w = weights[np.argmax(scores)]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки классов.

        Parameters
        ----------
        X : np.ndarray
            Набор данных, для которого необходимо вернуть метки классов.

        Return
        ------
        labels : np.ndarray
            Вектор индексов классов
            (по одной метке для каждого элемента из X).

        """
        X = np.column_stack((np.ones_like(X[:, 0]), X))
        y_pred = np.sign(X @ self.w.T).astype(int)
        y_pred = np.select([y_pred == -1, y_pred == 1], [self.label_1, self.label_2])

        return y_pred


X, true_labels = make_blobs(200, 2, centers=[[0, 0], [2.5, 2.5]])
c = Perceptron()
print(true_labels)
c.fit(X, true_labels)
y_pred = c.predict(X)
print(true_labels)
print(y_pred)
visualize(X, true_labels, np.array(y_pred), c.w)


