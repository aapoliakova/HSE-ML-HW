from sklearn.datasets import make_blobs, make_moons
import numpy as np
import pandas
import random
import matplotlib.pyplot as plt
import matplotlib
from typing import Callable, Union, NoReturn, Optional, Dict, Any, List


samples = [10, 0, 50]
x = np.array([0 for _ in range(10)] + [1 for _ in range(5)] + [1 for _ in range(45)])  # Метки классов
np.random.shuffle(x)


def gini(x: np.ndarray) -> float:
    """
    Считает коэффициент Джини для массива меток x.
    """
    _, n_samples = np.unique(x, return_counts=True)
    prob = n_samples / x.shape[0]
    return np.dot(prob, 1 - prob)


def entropy(x: np.ndarray) -> float:
    """
    Считает энтропию для массива меток x.
    """
    _, n_samples = np.unique(x, return_counts=True)
    prob = n_samples / x.shape[0]
    return -np.dot(prob, np.log2(prob))


def gain(left_y: np.ndarray, right_y: np.ndarray, criterion: Callable) -> float:
    """
    Считает информативность разбиения массива меток.

    Parameters
    ----------
    left_y : np.ndarray
        Левая часть разбиения.
    right_y : np.ndarray
        Правая часть разбиения.
    criterion : Callable
        Критерий разбиения.
    """
    x_node = np.concatenate([left_y, right_y])
    criterion = np.array([criterion(x_node), criterion(right_y), criterion(left_y)])
    prob = np.array([x_node.shape[0], -right_y.shape[0], -left_y.shape[0]])
    return np.dot(prob, criterion)


y_right = np.array([0 for _ in range(10)] + [1 for _ in range(5)] + [2 for _ in range(45)])  # Метки классов
y_left = np.array([0 for _ in range(10)] + [1 for _ in range(5)] + [1 for _ in range(45)])  # Метки классов

a = gain(y_left, y_right, criterion=gini)

if __name__ == "__main__":
    gain()


