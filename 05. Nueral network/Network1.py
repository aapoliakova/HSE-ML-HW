import numpy as np


class Module:
    """
    Абстрактный класс. Его менять не нужно.
    """

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, d):
        raise NotImplementedError()

    def update(self, alpha):
        pass


class Linear(Module):  # Это чисто один слой, то есть итерироваться по слоям не нужно уже
    """
    Линейный полносвязный слой.
    """

    def __init__(self, in_features: int, out_features: int):
        """
        Parameters
        ----------
        in_features : int
            Размер входа.
        out_features : int
            Размер выхода.

        Notes
        -----
        W и b инициализируются случайно.
        """
        scale = 1 / np.sqrt(in_features)
        self.weights = np.random.uniform(-scale, scale,
                                         (out_features, in_features))
        self.biases = np.zeros((1, out_features))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Возвращает a = Wx + b.

        Parameters
        ----------
        x : np.ndarray
            Входной вектор или батч.
            То есть, либо x вектор с in_features элементов,
            либо матрица размерности (batch_size, in_features).
            либо матрица размерности (in_features, batch_size)*

        Return
        ------
        a : np.ndarray
            Выход после слоя.
            Либо вектор с out_features элементами,
            либо матрица размерности (batch_size, out_features)

        """
        activation = np.dot(self.weights, x.T) + self.biases
        return activation

    def backward(self, d: np.ndarray) -> np.ndarray:
        """
        Cчитает градиент при помощи обратного распространения ошибки.

        Parameters
        ----------
        d : np.ndarray
            Градиент.
        Return
        ------
        np.ndarray
            Новое значение градиента.
        """



    def update(self, alpha: float) -> NoReturn:
        """
        Обновляет W и b с заданной скоростью обучения.

        Parameters
        ----------
        alpha : float
            Скорость обучения.
        """
