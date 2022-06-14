import numpy as np
import matplotlib.pyplot as plt

class GaussianBasisFunction:
    """
    f(x) = exp(-0.5* sum((x - mu)^2) / sigma^2)
    """
    def __init__(self, mu, sigma):
        """
        initialize Gaussian Basis Function
        :param mu: (K, D) or (K,)
        :param sigma:  float
        """
        if mu.ndim == 1:
            mu = mu[:, None] # mu.shape = (K, 1)
        self.mu = mu
        self.sigma = sigma

    def gaussian(self, x):
        muDim = self.mu.shape[1]

        if isinstance(x, float):
            assert muDim == 1

        if isinstance(x, np.ndarray):
            D = x.shape[0]
            assert muDim == D

        ret = []
        for i in range(len(self.mu)):
            ret.append(np.exp(-0.5 * np.sum(np.square(x - self.mu[i])) / self.sigma))
        ret = np.array(ret)
        return ret

    def transform(self, X):
        """
        transform X to phi(X)
        :param X:   (N, D) or (N,)
        :return:
        """
        if X.ndim == 1:
            X = X[:,None] # X.shape = (N, 1)
        N = X.shape[0]
        f = []
        for i in range(N):
            f.append(self.gaussian(X[i]))
        f = np.array(f)
        return f


