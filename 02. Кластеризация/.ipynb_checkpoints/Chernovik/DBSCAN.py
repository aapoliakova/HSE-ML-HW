import numpy as np
from sklearn.neighbors import KDTree
from collections import deque

X = np.array([

    [0, 0],
    [1, 1],
    [1, 0],
    [0, 1],
    [10, 10],
    [11, 10],
    [10, 11],
    [11, 11],
    [111, 111],
    [122, 122]
])


class DBScan:
    def __init__(self, eps: float = 0.5, min_samples: int = 5,
                 leaf_size: int = 40, metric: str = "euclidean"):
        """

        Parameters
        ----------
        eps : float, min_samples : int
            Параметры для определения core samples.
            Core samples --- элементы, у которых в eps-окрестности есть
            хотя бы min_samples других точек.
        metric : str
            Метрика, используемая для вычисления расстояния между двумя точками.
            Один из трех вариантов:
            1. euclidean
            2. manhattan
            3. chebyshev
        leaf_size : int
            Минимальный размер листа для KDTree.

        """
        self.eps = eps
        self.min_samples = min_samples
        self.leaf_size = leaf_size
        self.metric = metric

    def fit_predict(self, X: np.array, y=None) -> np.array:
        """
        Кластеризует элементы из X,
        для каждого возвращает индекс соотв. кластера.
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit_predict обязаны принимать
            параметры X и y, даже если y не используется).
        Return
        ------
        labels : np.array
            Вектор индексов кластеров
            (Для каждой точки из X индекс соотв. кластера).

        """
        kd_tree = KDTree(X, leaf_size=self.leaf_size, metric=self.metric)

        all_neighbors = kd_tree.query_radius(X, self.eps)
        labels, cluster = -np.ones(X.shape[0]), 0

        for idx, point in enumerate(X):
            if labels[idx] != -1:
                continue

            neighs = deque(all_neighbors[idx])

            if len(neighs) <= self.min_samples:
                continue

            # wait, thats a bfs!!
            labels[idx] = cluster

            while neighs:
                neigh = neighs.pop()

                if neigh == idx or labels[neigh] != -1:
                    continue

                labels[neigh] = cluster
                new_neighs = all_neighbors[neigh]

                if new_neighs.shape[0] > self.min_samples:
                    neighs.extendleft(new_neighs)

            cluster = cluster + 1

        labels[labels == -1] = cluster

        return labels

test = DBScan(min_samples=3, eps=1.5)
test.fit_predict(X)
