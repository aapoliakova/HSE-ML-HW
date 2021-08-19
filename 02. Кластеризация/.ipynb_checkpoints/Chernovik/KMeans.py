from sklearn.neighbors import KDTree
from sklearn.datasets import make_blobs, make_moons, make_swiss_roll
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
from collections import deque
from typing import NoReturn
from scipy.spatial import distance_matrix
import seaborn as sns

X, true_labels = make_blobs(400, 2, centers=[[0, 0], [-4, 0], [3.5, 3.5], [3.5, -2.0]], random_state=42)

def visualize_it(algorithm, n_dataset = 1):
    if n_dataset == 1:
        X, true_labels = make_blobs(400, 2, centers=[[0, 0], [-4, 0], [3.5, 3.5], [3.5, -2.0]], random_state=42)
        n_clusters = 4
        dbscan_eps, min_samples = 0.5, 1
    elif n_dataset == 2:
        X, true_labels = make_moons(400, noise=0.075, random_state=42)
        n_clusters = 2
        dbscan_eps, min_samples = 0.2, 5

    if algorithm == 'KMeans':
        data = {'sample': None, 'random': None, 'k-means++': None}
        for method in data.keys():
            data[method] = KMeans(n_clusters=n_clusters, init = method).fit(X).predict(X)
    elif algorithm == 'DBScan':
        data = {'euclidean': None, 'manhattan': None, 'chebyshev': None}
        for method in data.keys():
            data[method] = DBScan(metric = method, eps=dbscan_eps, min_samples = min_samples).fit_predict(X)

    elif algorithm == 'AgglomertiveClustering':
        data = {'average': None, 'single': None, 'complete': None}
        for method in data.keys():
            data[method] = AgglomertiveClustering(n_clusters=n_clusters, linkage = method).fit_predict(X)

    fig, axis = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    fig.suptitle(f"{algorithm}")
    sns.scatterplot(ax = axis[0], x = X[:, 0], y = X[:, 1], s = 100, hue = true_labels).set_title('True labels')

    for i in range(1, 4):
        method, labels = data.popitem()
        sns.scatterplot(ax = axis[i], x = X[:, 0], y = X[:, 1], s=100, hue=labels).set_title(method)

def euclidean_distance(X: np.array, centroids: np.array, sqrt: bool = True):
    """
    Considering the rows of centroids as vectors, compute the
    distance matrix between each pair of centroid and observation X (rows)

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    centroids : ndarray of shape (n_samples, n_clusters)
    sqrt : bool if False return squared distances

    Return
    ----------
    distances : ndarray of shape (n_samples, n_clusters)

    """
    if X[0].shape == centroids.shape:
        distances = np.linalg.norm(X - centroids, axis=1)[:, np.newaxis]
    else:
        n_samples, _ = X.shape
        distances = np.array([np.linalg.norm(X - center, axis=1)
                              for center in centroids]).transpose()

    if not sqrt:
        return distances ** 2
    return distances

class KMeans:
    """K-Means clustering.

    Parameters
    ----------
    n_clusters : int
        number of clusters to form
    init : str
        Method for initialization:
        1. random --- centroids initialized as random points,
        2. sample --- centroids are chosen at random from data ,
        3. k-means++ --- initial centroids selected using K-means++ methods.
    max_iter : int
        Maximum number of kmeans iterations.
    tol : float
        Tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
    verbose : bool
        Verbose mode

    """

    def __init__(self, n_clusters: int, init: str,
                 max_iter: int = 600, tol: float = 1e-4, verbose: bool = False):
        self.init = init
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def _init_centroids(self, X):
        """
        Computational component for initialization of n_clusters by
        k-means++.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Return
        ----------
        centroids : ndarray of shape (n_samples, n_clusters)
            The initial centers for k-means.
        """
        n_samples, n_features = X.shape
        centroids = np.empty((self.n_clusters, n_features), dtype=X.dtype)
        if self.init == 'kmeans++':

            index_id = np.random.randint(n_samples)
            centroids[0] = X[index_id]

            for c in range(1, self.n_clusters):
                closest_dist = np.min(euclidean_distance(X, centroids[:c, :], sqrt=False), axis=1)
                pick_random = np.random.random() * np.sum(closest_dist)
                cum_sum_dist = np.cumsum(closest_dist)
                cent_id = np.searchsorted(np.cumsum(closest_dist), v=pick_random, side='right')
                cent_id = np.clip(cent_id, 0, cum_sum_dist.size - 1)
                centroids[c] = X[cent_id]

        elif self.init == 'sample':
            indexes_id = np.random.permutation(n_samples)[:self.n_clusters]
            centroids = X[indexes_id]

        elif self.init == 'random':
            centroids = np.random.random((self.n_clusters, n_features))
        return centroids

    def fit(self, X: np.array, y=None) -> NoReturn:
        """Compute k-means clustering.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored

        Returns
        -------
        self
            Fitted estimator.
        """
        n_samples, n_features = X.shape
        centroids_new = self._init_centroids(X)
        labels_new = np.argmin(euclidean_distance(X, centroids_new), axis=1)

        while self.max_iter:
            self.max_iter -= 1
            centroids_old = centroids_new
            labels_old = labels_new

            centroids_new = np.empty((self.n_clusters, n_features), dtype=X.dtype)
            for c in range(self.n_clusters):
                indexes = np.where(labels_old == c)
                centroids_new[c] = np.mean(X[indexes], axis=0)

            labels_new = np.argmin(euclidean_distance(X, centroids_new), axis=1)

            # Check tol
            if np.array_equal(labels_old, labels_new):
                if self.verbose:
                    print(f"Converged at iteration {self.max_iter}: strict convergence.")
                break
            else:
                center_shift = np.linalg.norm(centroids_old - centroids_new, axis=1)
                center_shift_tot = (center_shift ** 2).sum()
                if center_shift_tot <= self.tol:
                    if self.verbose:
                        print(
                            f"Converged at iteration {self.max_iter}: center shift "
                            f"{center_shift_tot} within tolerance {self.tol}."
                        )
                break
        if self.verbose:
            print("Maximum of iteration reached")

        self.labels_ = labels_new
        self.centroids_ = centroids_new

        return self

    def predict(self, X: np.array, y=None) -> np.array:
        """Predict the closest cluster each sample in X belongs to.
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.fit(X).labels_


random_test = KMeans(n_clusters=4, init='random')
random_test.fit(X)

