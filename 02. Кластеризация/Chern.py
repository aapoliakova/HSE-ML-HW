#%%


# class KMeans:
#     def __init__(self, n_clusters: int, init: str = "random",
#                  max_iter: int = 300):

#         self.max_iter = max_iter
#         self.init = init
#         self.n_clusters = n_clusters
#         self.centroids = np.array([])

#     def fit(self, X: np.array, y=None) -> NoReturn:
#         """
#         Ищет и запоминает в self.centroids центроиды кластеров для X.

#         Parameters
#         ----------
#         X : np.array
#             Набор данных, который необходимо кластеризовать.
#         y : Ignored
#             Не используемый параметр, аналогично sklearn
#             (в sklearn считается, что все функции fit обязаны принимать
#             параметры X и y, даже если y не используется).

#         """


#         if self.init == 'k-means++':

#             centroids_init = np.array(random.choices(X))


#             for _ in range(self.n_clusters - 1):

#                 dist_to_closest = np.array([ # посчитали расстояние до этого центра
#                     np.min([np.linalg.norm(point - centroid) for centroid in centroids_init])
#                                      for point in X])


#                 cumulative_prob = np.cumsum(dist_to_closest/np.sum(dist_to_closest))
#                 selected_index = next(index for index, val in enumerate(cumulative_prob)
#                                                   if val > random.random())

#                 centroids_init = np.append(centroids_init, [X[selected_index]], axis=0)


#         elif self.init == 'random':
#             centroids_init = np.random.rand(self.n_clusters, len(X[0]))

#         elif self.init == 'sample':
#             centroids_init = random.choices(X, weights=None, cum_weights=None, k=self.n_clusters)


#         for _ in range(self.max_iter):

#             closest_centers = np.array([np.argmin(
#                 [np.linalg.norm(point - center)
#                         for center in centroids_init])
#                                 for point in X])

#             self.centroids = [np.mean(
#                     np.array([X[i] for i in range(len(X))
#                               if closest_centers[i] == j]), axis=0)
#                                                     for j in range(self.n_clusters)]
#             centroids_init = self.centroids

#     def predict(self, X: np.array) -> np.array:
#         """
#         Для каждого элемента из X возвращает номер кластера,
#         к которому относится данный элемент.

#         Parameters
#         ----------
#         X : np.array
#             Набор данных, для элементов которого находятся ближайшие кластера.

#         Return
#         ------
#         labels : np.array
#             Вектор индексов ближайших кластеров
#             (по одному индексу для каждого элемента из X).
#         """

#         labels = [np.argmin(
#             [np.linalg.norm(point - center)
#                             for center in self.centroids])
#                                     for point in X]
#         return np.array(labels)
X = np.array([
    [0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5],
    [5, 5], [4, 4], [4, 5], [5, 4], [4.5, 4.5],
    [15, 15], [15, 0], [3, 1], [7,5], [2.5, 2.5]
])