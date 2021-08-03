import numpy as np
y = np.array([5, 4, 3, 2, 1])
a = np.arange(25).reshape((5, 5))
x = np.full_like(y, 1)
print(x)

c = np.insert(X, 0, np.full_like(y, 1), axis = 1)
print(c)
