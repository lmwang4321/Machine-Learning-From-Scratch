import numpy as np
import itertools
import functools
import matplotlib.pyplot as plt

class PolynomialBasisFunction:
    """
    Let x = [a, b]
    phi(x, degree=2) = [1, a, b, a^2, ab, b^2]
    """
    def __init__(self, degree):
        self.degree = degree

    def transform(self, X):
        """
        X.shape = (N,) or (N, D)
        :param X:
        :return:
        """
        if X.ndim == 1:
            X = X[:, None] # X.shape = (N, 1)
        X_t = X.T

        features = [np.ones(X.shape[0])]
        for d in range(1, self.degree+1):
            for item in itertools.combinations_with_replacement(X_t, d):
                features.append(functools.reduce(lambda x, y: x*y, item))
        features = np.array(features).T
        return features

