import enum
import matplotlib.pyplot as plt
import numpy as np
import os
import itertools

def count_freq(x, y, net_N, lim):

    xlim, ylim = lim
    net_X, net_Y = net_N
    x_tick, y_tick = xlim / net_X, ylim / net_Y
    x_band = np.floor(x / x_tick).reshape(-1,1)
    y_band = np.floor(y / y_tick).reshape(-1,1)
    _, idx, count = np.unique(np.concatenate((x_band, y_band), axis=1), axis=0, return_counts=True, return_index=True)

    coef = np.zeros(net_N)
    xmesh =  ymesh = np.arange(net_X)
    for graph_loc in itertools.product(xmesh[:-1], ymesh[:-1]):
        x_loc, y_loc = graph_loc
        coef_mask = ((x_band == x_loc).flatten() * (y_band == y_loc).flatten()).nonzero()[0]
        coef_x, coef_y = x[coef_mask], y[coef_mask]
        coef_loc = np.corrcoef(coef_x, coef_y)[0,1]
        coef[y_loc, x_loc] = coef_loc if np.isnan(coef_loc) == False else 0.0


    return count.reshape(net_X, net_Y)

output_dir = "/media/pjh3974/demo/test"

net = (10, 10)
lim = (1.0, 1.0)
test_x = np.random.rand(1000)
test_y = np.random.rand(1000)
freq = count_freq(test_x, test_y, net, lim)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))


X, Y = (np.linspace(0,lim_n,net_n+1) for lim_n, net_n in zip(lim, net))
ax[0].pcolor(X, Y, freq, alpha=0.4, cmap="BuGn", shading='auto')
ax[0].scatter(test_x, test_y)

for axis in ax.flatten():
    axis.set_xlim(0,1)
    axis.set_ylim(0,1)
    axis.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "graph.png"))
