import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston

# cargamos el dataset
boston = load_boston()

print(boston.DESCR)

X = np.array(boston.data[:, 5])
Y = np.array(boston.target)


plt.scatter(X, Y, alpha=0.3)

# anadir columna de unos para termino independiente
X = np.c_[np.ones(X.shape[0]), X]
print(X)

# Matriz de parametros que minimiza los errores cuadraticos medios
B= np.linalg.inv(X.T @ X) @ X.T @ Y
print(B)

plt.plot([4,9],[B[0] + B[1]*4, B[0] + B[1]*9], color='red')

plt.show()

