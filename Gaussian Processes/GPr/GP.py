"""
author: Chao Wang
Institution: Dalhousie University, Faculty of Computer Science
"""
from kernel.SquaredExponential import SquaredExponential as SE
import numpy as np
from numpy import linalg as la

class GPRegression:
    def __init__(self, sigma=None, l=None):
        # initialize kernel parameters
        self.sigma = sigma
        self.l = l

    def predict(self, X_train, X_test, y):
        K = SE(X_train)
        n = K.shape[0]
        L = la.cholesky(K + self.sigma ** 2 * np.eye(n))
        alpha = la.solve(L.T, la.solve(L, y))
        kStar = SE(X_train, X_test)
        fStar_bar = kStar.T@alpha
        v =