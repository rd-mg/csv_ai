import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

func = lambda th: np.sin(1/2 * th[0]**2 -1/4 * th[1]**2 + 3) *  np.cos(2* th[0] + 1 -np.e ** th[1])

res = 100
_X = np.linspace(-2, 2, res)
_Y = np.linspace(-2, 2, res)
_Z = np.zeros((res, res))

for ix, x in enumerate(_X):
    for iy, y in enumerate(_Y):
        _Z[ix, iy] = func([x, y])

plt.contourf(_X, _Y, _Z, 100)
plt.colorbar()


Theta = np.random.rand(2) * 4 - 2
plt.plot(Theta[0],Theta[1], 'o', c='red')
plt.show()
