import numpy as np
import matplotlib.pyplot as plt

def generate_PSD_mat(D:int)->np.ndarray:

    A = np.random.randn(D, D) * 0.1
    return A.T@A

def plot_contourf(data, func, lines=3):
    n = 256
    x = np.linspace(data[:, 0].min(), data[:, 0].max(), n)
    y = np.linspace(data[:, 1].min(), data[:, 1].max(), n)
    X, Y = np.meshgrid(x, y)
    C = plt.contour(X, Y, func(np.c_[X.reshape(-1), Y.reshape(-1)]).reshape(X.shape), lines, colors='g', linewidth=0.5)
    # C = plt.contour(X, Y, func(np.c_[X.reshape(-1), Y.reshape(-1)]).reshape(X.shape), cmap = cm.jet, alpha = 0.5, extent = [1, -1, 1, -1])
    # plt.imshow(func(np.c_[X.reshape(-1), Y.reshape(-1)]).reshape(X.shape)[:, ::-1])
    plt.clabel(C, inline=True, fontsize=10)
    plt.scatter(data[:, 0], data[:, 1])
