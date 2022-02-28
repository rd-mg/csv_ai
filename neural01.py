import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles

#CREAR EL DATASET
n = 500
p = 2

X, y = make_circles(n_samples=n, factor=0.5, noise=0.05)

plt.scatter(X[y==0,0], X[y==0,1], c='yellow')
plt.scatter(X[y==1,0], X[y==1,1], c='green')
plt.show()

