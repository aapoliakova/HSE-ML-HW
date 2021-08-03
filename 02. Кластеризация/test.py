from sklearn.neighbors import KDTree
from sklearn.datasets import make_blobs, make_moons, make_swiss_roll
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
# import opencv
from collections import deque
from typing import NoReturn

X = np.array([
    [4, 1], [1, 5], [1, 1], [4, 5],
    [14, 11], [11, 15], [11, 11], [14, 15],
    [24, 21], [21, 25], [21, 21], [24, 25],

])
true_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])


#
# # Считаем матрицу расстояний
# eps = 100
# dist = np.zeros((len(X), len(X)))
#
# for i in range(len(X)):
#     for j in range(len(X)):
#         dist[i, j] = np.int(np.linalg.norm(X[i] - X[j]))
#         dist[i, i] = eps
#
#
# # single  - берем минимальное расстояние между точками в двух разных кластерах
#
# # 1. Инициализируем все точки как отдельные кластеры, соединяем те, между которыми минимальное расстояние.
# # В матрице расстояний, пересчитываем новое расстояние до точки как до ближайшей точки принадлежащей кластеру
# # Меняем номер кластера на первый индекс
#
# labels = (np.arange(len(X)))
# len_labels = len(set(labels))
# n_clusters = 3
# max_iter = 0
#
# while len_labels > n_clusters:
#     i, j = np.unravel_index(np.argmin(dist), dist.shape)
#     labels[labels == j] = labels[i]  # Проверить
#     dist[i, j] = dist[j, i] = eps
#     for el in labels:
#         if el != i:
#             print(el, (i, j))
#             print()
#             dist[i, el] = dist[el, i] = dist[el, j] = dist[j, el] = min(dist[i, el], dist[j, el])
#             print(dist)
#             print(labels)
#             print()
#
#     len_labels = len(set(labels))
#     max_iter += 1
# print(max_iter)


class AgglomertiveClustering:
    def __init__(self, n_clusters: int = 16, linkage: str = "average"):
        """

        Parameters
        ----------
        n_clusters : int
            Количество кластеров, которые необходимо найти (то есть, кластеры
            итеративно объединяются, пока их не станет n_clusters)
        linkage : str
            Способ для расчета расстояния между кластерами. Один из 3 вариантов:
            1. average --- среднее расстояние между всеми парами точек,
               где одна принадлежит первому кластеру, а другая - второму.
            2. single --- минимальное из расстояний между всеми парами точек,
               где одна принадлежит первому кластеру, а другая - второму.
            3. complete --- максимальное из расстояний между всеми парами точек,
               где одна принадлежит первому кластеру, а другая - второму.
        """
        self.n_clusters = n_clusters
        self.linkage = linkage

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
        self.size = len(X)
        eps = 1e8
        dist = np.zeros((self.size, self.size))

        for i in range(self.size):
            for j in range(self.size):
                dist[i, j] = np.linalg.norm(X[i] - X[j])
                dist[i, i] = eps

        labels = np.arange(self.size)
        curr_n_cluster = len(set(labels))

        while curr_n_cluster != self.n_clusters:
            i, j = np.unravel_index(np.argmin(dist), dist.shape)
            dist[i, j] = dist[j, i] = eps

            labels[labels == j] = labels[i]

            if self.linkage == "average":
                for el in labels:
                    if el != i:
                        dist[i, el] = dist[el, i] = dist[el, j] = dist[j, el] = np.mean([dist[i, el], dist[j, el]])

            elif self.linkage == "single":
                for el in labels:
                    if el != i:
                        dist[i, el] = dist[el, i] = dist[el, j] = dist[j, el] = min(dist[i, el], dist[j, el])

            elif self.linkage == "complete":
                for el in labels:
                    if el != i:
                        dist[i, el] = dist[el, i] = dist[el, j] = dist[j, el] = max(dist[i, el], dist[j, el])
            curr_n_cluster = len(set(labels))

        tmp = list(set(labels))

        for i in range(len(tmp)):
            labels[labels == tmp[i]] = i

        return labels

# X_1, true_labels = make_blobs(200, 2, centers=[[0, 0], [-4, 0], [3.5, 3.5], [3.5, -2.0]])
X_1, true_labels2 = make_moons(300, noise=0.075)


ggcluster = AgglomertiveClustering(n_clusters=4, linkage="complete")
labels = ggcluster.fit_predict(X_1)
print(len(set(labels)))
plt.scatter(X_1[:, 0], X_1[:, 1], c=labels)
plt.show()
print(labels)
