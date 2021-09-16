from sklearn.datasets import make_blobs, make_moons
import numpy as np
import pandas
import random
import matplotlib.pyplot as plt
import matplotlib
from typing import Callable, Union, NoReturn, Optional, Dict, Any, List
import os
from Task1 import gain, gini, entropy


class DecisionTreeLeaf:
    """

    Attributes
    ----------
    y : int
        return index of majority class
    probabilities : dict
        Словарь, отображающий метки в вероятность того, что объект, попавший в данный лист, принадлжит классу, соответствующиему метке

    x : набор меток классов для каждой точки - на вход джини  [1, 1, 0, 2, 0 ,0, 2]
        Словарь, отображающий метки в вероятность того, что объект, попавший в данный лист, принадлжит классу, соответствующиему метке
    """

    def __init__(self, y):
        self.classes, self.samples = np.unique(y, return_counts=True)
        self.y = np.argmax(self.samples)
        self.probabilities = dict(zip(y, self.samples / y.size))


class DecisionTreeNode:
    """

    Attributes
    ----------
    split_dim : int
        Измерение, по которому разбиваем выборку.
    split_value : float
        Значение, по которому разбираем выборку.
    left : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] < split_value.
    right : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] >= split_value.
    """

    def __init__(self, split_dim: int, split_value: float,
                 left: Union['DecisionTreeNode', DecisionTreeLeaf],
                 right: Union['DecisionTreeNode', DecisionTreeLeaf]):
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right

    def walk_tree(self, x):
        if x[self.split_dim] < self.split_value:
            if isinstance(self.left, DecisionTreeLeaf):
                return self.left.probabilities
            return self.left.walk_tree(x)
        if isinstance(self.right, DecisionTreeLeaf):
            return self.right.probabilities
        return self.right.walk_tree(x)


class DecisionTreeClassifier:
    """
    Attributes
    ----------
    root : Union[DecisionTreeNode, DecisionTreeLeaf]
        Корень дерева.

    (можете добавлять в класс другие аттрибуты).

    """

    def __init__(self, criterion: str = "gini",
                 max_depth: Optional[int] = None,
                 min_samples_leaf: int = 1):
        """
        Parameters
        ----------
        criterion : str
            Задает критерий, который будет использоваться при построении дерева.
            Возможные значения: "gini", "entropy".
        max_depth : Optional[int]
            Ограничение глубины дерева. Если None - глубина не ограничена.
        min_samples_leaf : int
            Минимальное количество элементов в каждом листе дерева.

        """
        criterions = {'gini': gini, 'entropy': entropy}

        self.criterion = criterions[criterion]
        self.root = None
        self.max_depth = max_depth
        self.min_leaf = min_samples_leaf

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Строит дерево решений по обучающей выборке.

        Parameters
        ----------
        X : np.ndarray
            Обучающая выборка.
        y : np.ndarray
            Вектор меток классов.
        """
        self.root = self.build_node(X, y)

    def build_node(self, X: np.ndarray, y: np.ndarray, depth: int = 0):
        if self.max_depth is not None and depth >= self.max_depth:
            return DecisionTreeLeaf(y)
        else:
            split_val, split_dim, max_gain = None, None, 0
            n_samples, n_features = X.shape
            for feature in range(n_features):
                for val in X[:, feature]:
                    mask = X[:, feature] < val

                    if mask.sum() >= self.min_leaf and n_samples - mask.sum() >= self.min_leaf:
                        info_gain = gain(y[mask], y[~mask], criterion=self.criterion)
                        if info_gain > max_gain:
                            split_val, split_dim, max_gain = val, feature, info_gain
            if max_gain == 0:
                return DecisionTreeLeaf(y)
            else:
                mask = X[:, split_dim] < split_val
                left = self.build_node(X[mask], y[mask], depth + 1)
                right = self.build_node(X[~mask], y[~mask], depth + 1)
                return DecisionTreeNode(split_dim, split_val, left, right)

    def predict_proba(self, X: np.ndarray) -> List[Dict[Any, float]]:
        """
        Предсказывает вероятность классов для элементов из X.

        Parameters
        ----------
        X : np.ndarray
            Элементы для предсказания.

        Return
        ------
        List[Dict[Any, float]]
            Для каждого элемента из X возвращает словарь
            {метка класса -> вероятность класса}.
        """
        prediction = []
        for x in X:
            if isinstance(self.root, DecisionTreeLeaf):
                prediction.append(self.root.probabilities)
            else:
                prediction.append(self.root.walk_tree(x))
        return prediction

    def predict(self, X: np.ndarray) -> list:
        """
        Предсказывает классы для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Элементы для предсказания.

        Return
        ------
        list
            Вектор предсказанных меток для элементов X.
        """
        proba = self.predict_proba(X)
        return [max(p.keys(), key=lambda k: p[k]) for p in proba]


X, y = make_moons(200, noise=0.35, random_state=42)



cls = DecisionTreeClassifier()
cls.fit(X, y)
# print(type(cls.root))
# print(type(cls.root.left.left.left.y))
# print(cls.root.left.left.left.probabilities)
print(cls.predict(X))
