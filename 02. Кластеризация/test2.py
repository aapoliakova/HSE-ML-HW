import matplotlib.pyplot as plt
import numpy as np

X = np.array([
    [4, 1], [1, 5], [1, 1], [4, 5],
    [14, 11], [11, 15], [11, 11], [14, 15],
    [24, 21], [21, 25], [21, 21], [24, 25],

])


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
        if linkage == "average":
            self.f = lambda a, b: np.mean([a, b], axis=0)
        if linkage == "single":
            self.f = min
        if linkage == "complete":
            self.f = max
        self.eps = 100

    def calc_dist(self, X, points):
        dist = np.array([np.sum((X - y) ** 2, axis=1)**(1/2) for y in points])
        for i in range(X.shape[0]):
            dist[i, i] = self.eps
        return dist

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

        temporary_classes = np.arange(X.shape[0])
        dist_matrix = self.calc_dist(X, X)

        clusters = list(set(temporary_classes))
        while len(clusters) > self.n_clusters:
            cl1, cl2 = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            print(cl1, cl2)

            for cl in clusters:
                if cl == cl1:
                    continue
                dist_matrix[cl, cl1] = dist_matrix[cl1, cl] = self.f(dist_matrix[cl1, cl], dist_matrix[cl2, cl])
            print(dist_matrix)

            temporary_classes[temporary_classes == cl2] = cl1
            dist_matrix[cl2] = dist_matrix[:, cl2] = self.eps
            clusters = list(set(temporary_classes))

        for i in range(len(clusters)):
            temporary_classes[temporary_classes == clusters[i]] = i

        return temporary_classes


agg_clustering = AgglomertiveClustering(n_clusters=4, linkage="single")
labels = agg_clustering.fit_predict(X)

