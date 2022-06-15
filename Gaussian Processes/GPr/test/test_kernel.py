import os
os.chdir("..")

import numpy as np
from kernel.SquaredExponential import SquaredExponential as SE
# f = SquaredExponential1D(x1)

def kernel(a, b):
    kernelParameter_l = 0.1
    kernelParameter_sigma = 1.0
    sqdist = np.sum(a**2,axis=1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    # np.sum( ,axis=1) means adding all elements columnly; .reshap(-1, 1) add one dimension to make (n,) become (n,1)
    return kernelParameter_sigma*np.exp(-.5 * (1/kernelParameter_l) * sqdist)

#testing K(X)'s shape
def test1():
    X = np.array([1, 2, 3, 4, 5])
    K = SE(X)
    assert K.shape == (5, 5)

# testing K(X, X*)'s shape
def test2():
    X_train = np.array([1, 2, 3, 4, 5])
    X_test = np.array([1])
    K = SE(X_train, X_test)
    assert K.shape == (5,1)

# testing result of K(X, X)
def test3():
    N = 20
    X = np.array([1, 2, 3, 4])
    Xc = X[:, None]
    K1 = kernel(Xc, Xc)
    K2 = SE(X, l=0.1, sigma=1.0)
    assert (K1 == K2).all()


test3()