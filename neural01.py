from audioop import reverse
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles

# CREAR EL DATASET
n = 500
p = 2

X, y = make_circles(n_samples=n, factor=0.5, noise=0.05)

y = y[:,np.newaxis]

# plt.scatter(X[y[:,0]==0,0], X[y[:,0]==0,1], c='yellow')
# plt.scatter(X[y[:,0]==1,0], X[y[:,0]==1,1], c='green')
# plt.show()

# CLASE DE LA CAPA DE LA RED
class neural_layer():
    def __init__(self, n_conn, n_neur, act_f) -> None:
        self.act_f = act_f
        self.b = np.random.rand(1, n_neur) * 2 -1 
        self.w = np.random.rand(n_conn, n_neur) * 2 -1

# FUNCIONES DE ACTIVACION
sigmoid = (lambda x: 1 / (1 + np.exp(-x)),
              lambda x: x * (1 - x))    

_x = np.linspace(-5, 5, 100)
# plt.plot(_x, sigmoid[0](_x))
# plt.show()

# CREAR LAS CAPAS DE LA RED
l0 = neural_layer(p, 4, sigmoid)
l1 = neural_layer(4, 2, sigmoid)

topology = [p,4,8,1]

def create_net(topology, act_f):
    net = []
    for i in range(len(topology)-1):
        net.append(neural_layer(topology[i], topology[i+1], act_f))
    return net

neural_net = create_net(topology, sigmoid)

l2_cost = (lambda Yp, Yr: np.mean((Yp-Yr)**2), lambda Yp , Yr : (Yp - Yr))

def train(neural_net, X, y, act_f, lr=0.05, train = True):
    out = [(None, X)]

    # forward pass
    for l, layer in enumerate(neural_net):
        z = out[-1][1] @ layer.w + layer.b
        a = layer.act_f[0](z)

        out.append((z, a))
    # print(act_f[0](out[-1][1],y))

    if train:
    # TRAINING
    # backward pass
        deltas = []
        for l in reversed(range(0,len(neural_net))):
            z = out[l+1][0]
            a = out[l+1][1]
            if l == len(neural_net)-1:
                deltas.insert(0, l2_cost[1](a, y) * neural_net[1].act_f[1](a))
            else:
                deltas.insert(0, deltas[0] @ _W.T * neural_net[1].act_f[1](a))
            _W = neural_net[l].w

            # Gradient descent
            neural_net[l].b = neural_net[l].b - np.mean(deltas[0], axis=0, keepdims=True) * lr
            neural_net[l].w = neural_net[l].w - out[l][1].T @ deltas[0] * lr
        
    return out[-1][1]


train(neural_net, X, y, l2_cost, 0.05, True)

# --------------------------------------------------
# PREDICCION

import time 
from IPython.display import clear_output

neural_n = create_net(topology, sigmoid)
loss = []

for i in range(1500):
    # TRAINING
    pY = train(neural_n, X, y, l2_cost)

    if i % 25 == 0:
        loss.append(l2_cost[0](pY, y))

        res = 50

        _x0 = np.linspace(-1.5, 1.5, res)
        _x1 = np.linspace(-1.5, 1.5, res)

        _y = np.zeros((res, res))

        for i0, x0 in enumerate(_x0):
            for i1, x1 in enumerate(_x1):
                _y[i0, i1] = train(neural_n, np.array([[x0, x1]]), y, l2_cost, train= False)[0][0]

    if i % 200 == 0:
        plt.pcolormesh(_x0, _x1, _y, cmap='RdBu_r')
        plt.axis('equal')

        plt.scatter(X[y[:,0]==0,0], X[y[:,0]==0,1], c='skyblue')
        plt.scatter(X[y[:,0]==1,0], X[y[:,0]==1,1], c='orange')

        clear_output(wait=True)
        # plt.show()
        plt.plot(range(len(loss)), loss)
        # plt.show()
        time.sleep(0.5)