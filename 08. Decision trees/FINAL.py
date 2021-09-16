import numpy as np
from Task1 import gini, entropy, gain


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


#  Добавить бэггинг

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


import pandas as pd

data = pd.read_csv("/Users/aapoliakova/PycharmProjects/ML/Homeworks/09. Random forest/vk.csv", header=1)
data = np.array(data.values.tolist())
X = np.array(data[:10, 2:], dtype=np.int)
y, y1 = data[:10, 0], data[:100, 1]

cls = DecisionTree(X, y, max_depth=5, max_features=5)
labels = cls.predict(X)
cls.predict(cls.out_of_bag_X)

print("Accuracy:", np.mean(labels == y))


cls2 = DecisionTree(X, y, max_depth=5, max_features=5)
labes2 = cls2.predict(X)

from scipy.stats import mode
#
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


def synthetic_dataset(size):
    X = [(np.random.randint(0, 2), np.random.randint(0, 2), i % 6 == 3,
          i % 6 == 0, i % 3 == 2, np.random.randint(0, 2)) for i in range(size)]
    y = [i % 3 for i in range(size)]
    return np.array(X), np.array(y)

X, y = synthetic_dataset(10)
rfc = RandomForestClassifier(n_estimators=5)
rfc.fit(X, y)
print("----------------------")
print(X.shape)

print("Accuracy:", np.mean(rfc.predict(X) == y))


def feature_importance(rfc):
    # Внешний цикл деревья, внутренний фичи
    n_trees, n_features = rfc.n_estimators, rfc.n_features

    importance_matrix = np.empty((n_features, n_trees), dtype=np.float)
    estimators = rfc.trees

    for tree_n in range(n_trees):
        tree = estimators[tree_n]
        X_out, y_out = tree.out_of_bag_X, tree.out_of_bag_y
        err_oob = np.mean(y_out == tree.predict(X_out))
        for feature in range(n_features):
            shuffled_out = X_out.copy()
            shuffled_out[:, feature] = np.random.permutation(X_out[:, feature])
            err_oob_sh = np.mean(y_out == tree.predict(shuffled_out))
            importance = err_oob - err_oob_sh
            importance_matrix[feature, tree_n] = importance
    return np.mean(importance_matrix, axis=1)

imp_m = feature_importance(rfc)
print(imp_m)
print(imp_m.shape)





feature_importance(rfc)

# def most_important_features(importance, names, k=20):
#     # Выводит названия k самых важных признаков
#     idicies = np.argsort(importance)[::-1][:k]
#     return np.array(names)[idicies]

