from sklearn.model_selection import train_test_split
import numpy as np
import pandas
import random
from scipy.stats import mode
import matplotlib.pyplot as plt
import matplotlib
import copy
from catboost import CatBoostClassifier
from tqdm import tqdm



def gini(x):
    _, counts = np.unique(x, return_counts=True)
    proba = counts / len(x)
    return np.sum(proba * (1 - proba))


def entropy(x):
    _, counts = np.unique(x, return_counts=True)
    proba = counts / len(x)
    return -np.sum(proba * np.log2(proba))


def gain(left_y, right_y, criterion):
    y = np.concatenate((left_y, right_y))
    return criterion(y) - (criterion(left_y) * len(left_y) + criterion(right_y) * len(right_y)) / len(y)


class Leaf:
    def __init__(self, y):
        self.y = y
        self.classes, self.samples = np.unique(y, return_counts=True)
        self.predicted = self.classes[np.argmax(self.samples)]


class Node:
    def __init__(self, split_dim, left, right):
        self.split_dim = split_dim
        self.left = left
        self.right = right


class DecisionTree:
    def __init__(self, X, y, criterion="gini", max_depth=None, min_samples_leaf=1, max_features="auto"):

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = entropy if criterion == 'entropy' else gini
        bagging_indexes = np.random.choice(X.shape[0], X.shape[0], replace=True)
        self.bagging_X, self.bagging_y = X[bagging_indexes], y[bagging_indexes]
        self.out_of_bag_X = np.delete(X, bagging_indexes, axis=0)
        self.out_of_bag_y = np.delete(y, bagging_indexes)



    def build_node(self, X, y, depth=0):

        if self.max_depth is not None and depth >= self.max_depth:
            return Leaf(y)

        n_samples, n_features = X.shape
        if self.max_features != 'auto':
            k = int(np.sqrt(n_features))
            features = np.random.choice(n_features, k, replace=False)
        else:
            features = np.arange(n_features)
        # As we have only binary features {0, 1} we split everything on feature == 1 and not
        split_dim, max_gain = None, 0.0
        for feature in features:
            mask = X[:, feature] == 0

            could_be_leaf = mask.sum() > self.min_samples_leaf and n_samples - mask.sum() > self.min_samples_leaf
            if could_be_leaf:
                info_gain = gain(y[mask], y[~mask], criterion=self.criterion)
                if info_gain > max_gain:
                    split_dim, max_gain = feature, info_gain
        if split_dim is None:
            return Leaf(y)

        mask = X[:, split_dim] == 0
        left = self.build_node(X[mask], y[mask], depth + 1)
        right = self.build_node(X[~mask], y[~mask], depth + 1)
        return Node(split_dim, left, right)

    def predict(self, X, node=None):
        result = np.empty(X.shape[0], dtype=np.object)

        if node is None:
            node = self.build_node(self.bagging_X, self.bagging_y)
        if isinstance(node, Leaf):
            result[:] = node.predicted
            return result
        else:
            mask = X[:, node.split_dim] == 0
            result[mask] = self.predict(X[mask], node.left)
            result[~mask] = self.predict(X[~mask], node.right)
            return result


class RandomForestClassifier:
    def __init__(self, criterion="gini", max_depth=None,
                 min_samples_leaf=1, max_features="auto", n_estimators=10):

        self.n_estimators = n_estimators
        self.trees = None
        self.params = {'criterion': criterion, 'max_depth': max_depth,
                       'min_samples_leaf': min_samples_leaf, 'max_features': max_features}



    def fit(self, X, y):
        self.trees = [DecisionTree(X, y, **self.params) for _ in range(self.n_estimators)]

    def predict(self, X):
        self.n_features = X.shape[1]
        predicted_trees = np.array([tree.predict(X) for tree in self.trees]).T
        hard_votes = np.array([mode(votes)[0][0] for votes in predicted_trees])
        return hard_votes


# def feature_importance(rfc):
#     # Внешний цикл деревья, внутренний фичи
#     n_trees, n_features = rfc.n_estimators, rfc.n_features
#
#     importance_matrix = np.empty((n_features, n_trees), dtype=np.float64)
#     estimators = rfc.trees
#
#     for tree_n in range(n_trees):
#         tree = estimators[tree_n]
#         X_out, y_out = tree.out_of_bag_X, tree.out_of_bag_y
#         err_oob = np.mean(y_out == tree.predict(X_out))
#         for feature in range(n_features):
#             print(tree_n, feature)
#             shuffled_out = np.copy(X_out)
#             shuffled_out[:, feature] = np.random.permutation(X_out[:, feature])
#             err_oob_sh = np.mean(y_out == tree.predict(shuffled_out))
#             importance_matrix[feature, tree_n] = err_oob - err_oob_sh
#
#     return np.mean(importance_matrix, axis=1)


# def feature_importance(rfc):
#
#     n_trees, n_features = rfc.n_estimators, rfc.n_features
#
#     importance_matrix = np.empty((n_features, n_trees), dtype=np.float64)
#     estimators = rfc.trees
#
#     with tqdm(total=n_trees * n_features) as pbar:
#
#         for tree_n in tqdm(range(n_trees)):
#             tree = estimators[tree_n]
#             X_out, y_out = tree.out_of_bag_X, tree.out_of_bag_y
#             err_oob = np.mean(y_out == tree.predict(X_out))
#             for feature in tqdm(range(n_features)):
#                 shuffled_out = np.copy(X_out)
#                 shuffled_out[:, feature] = np.random.permutation(X_out[:, feature])
#                 err_oob_sh = np.mean(y_out == tree.predict(shuffled_out))
#                 importance_matrix[feature, tree_n] = err_oob - err_oob_sh
#                 pbar.update(1)
#         return np.mean(importance_matrix, axis=1)





# def most_important_features(importance, names, k=5):
#     # Выводит названия k самых важных признаков
#     idicies = np.argsort(importance)[::-1][:k]
#     return np.array(names)[idicies]


def read_dataset(path):
    dataframe = pandas.read_csv(path, header=0)
    dataset = dataframe.values.tolist()
    random.shuffle(dataset)
    y_age = [row[0] for row in dataset]
    y_sex = [row[1] for row in dataset]
    X = [row[2:] for row in dataset]

    return np.array(X), np.array(y_age), np.array(y_sex), list(dataframe.columns)[2:]



X, y_age, y_sex, features = read_dataset("vk.csv")
X_train, X_test, y_age_train, \
y_age_test, y_sex_train, y_sex_test = train_test_split(X, y_age, y_sex, train_size=0.9)





# rfc = RandomForestClassifier(n_estimators=10)
# rfc.fit(X_train, y_age_train)
# print("Accuracy:", np.mean(rfc.predict(X_test) == y_age_test))
# names = feature_importance(rfc)


def synthetic_dataset(size):
    X = [(np.random.randint(0, 2), np.random.randint(0, 2), i % 6 == 3,
          i % 6 == 0, i % 3 == 2, np.random.randint(0, 2)) for i in range(size)]
    y = [i % 3 for i in range(size)]
    return np.array(X), np.array(y)

X, y = synthetic_dataset(1000)
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X, y)
print("Accuracy:", np.mean(rfc.predict(X) == y))
print("Importance:", *feature_importance(rfc))

