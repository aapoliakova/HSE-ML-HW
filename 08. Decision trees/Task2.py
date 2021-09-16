# from sklearn.datasets import make_blobs, make_moons
# import numpy as np
# import pandas
# import random
# import matplotlib.pyplot as plt
# import matplotlib
# from typing import Callable, Union, NoReturn, Optional, Dict, Any, List
# ### Задание 2 (1 балл)
# # Деревья решений имеют хорошую интерпретируемость, т.к. позволяют не только предсказать класс, но и объяснить,
# # почему мы предсказали именно его. Например, мы можем его нарисовать.
# # Чтобы сделать это, нам необходимо знать, как оно устроено внутри. Реализуйте классы, которые будут задавать структуру дерева.
#
# #### DecisionTreeLeaf
# # Поля:]
# # 1. `y` должно содержать класс, который встречается чаще всего среди элементов листа дерева
# #
# # #### DecisionTreeNode
# # В данной домашней работе мы ограничемся порядковыми и количественными признаками,
# # поэтому достаточно хранить измерение и значение признака, по которому разбиваем обучающую выборку.
# #
# # Поля:
# # 1. `split_dim` измерение, по которому разбиваем выборку
# # 2. `split_value` значение, по которому разбираем выборку
# # 3. `left` поддерево, отвечающее за случай `x[split_dim] < split_value`.
# # Может быть `DecisionTreeNode` или `DecisionTreeLeaf`
# # 4. `right` поддерево, отвечающее за случай `x[split_dim] >= split_value`.
# # Может быть `DecisionTreeNode` или `DecisionTreeLeaf`
# #
# # __Интерфейс классов можно и нужно менять при необходимости__
#
#
#
# class DecisionTreeLeaf:
#     """
#
#     Attributes
#     ----------
#     y : int
#         return index of majority class
#     probabilities : dict
#         Словарь, отображающий метки в вероятность того, что объект, попавший в данный лист, принадлжит классу, соответствующиему метке
#
#     x : набор меток классов для каждой точки - на вход джини  [1, 1, 0, 2, 0 ,0, 2]
#         Словарь, отображающий метки в вероятность того, что объект, попавший в данный лист, принадлжит классу, соответствующиему метке
#     """
#     def __init__(self, x: np.ndarray):
#         self.classes, self.samples = np.unique(x, return_counts=True)
#         self.y = np.argmax(self.samples)
#         self.probabilities = {self.classes[i]: self.samples[i]/x.shape[0]
#                               for i in range(x.shape[0])}
#
#
# class DecisionTreeNode:
#     """
#
#     Attributes
#     ----------
#     split_dim : int
#         Измерение, по которому разбиваем выборку.
#     split_value : float
#         Значение, по которому разбираем выборку.
#     left : Union[DecisionTreeNode, DecisionTreeLeaf]
#         Поддерево, отвечающее за случай x[split_dim] < split_value.
#     right : Union[DecisionTreeNode, DecisionTreeLeaf]
#         Поддерево, отвечающее за случай x[split_dim] >= split_value.
#     """
#     def __init__(self, split_dim: int, split_value: float,
#                  left: Union['DecisionTreeNode', DecisionTreeLeaf],
#                  right: Union['DecisionTreeNode', DecisionTreeLeaf]):
#         self.split_dim = split_dim
#         self.split_value = split_value
#         self.left = left
#         self.right = right
