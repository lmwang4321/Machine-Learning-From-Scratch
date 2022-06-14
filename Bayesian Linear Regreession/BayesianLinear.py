import matplotlib.axes
import numpy as np
import matplotlib.pyplot as plt
import utils
from scipy.stats import multivariate_normal

class BayesianLinear:
    """
    Bayesian Linear Regression
    """
    def __init__(self, alpha, beta):
        """

        :param alpha:
        :param beta:
        """
        self.alpha = alpha
        self.beta = beta

        self.mN = None
        self.SN = None

    def init_params(self, n_dim):
        """
        prior parameters for p(w|alpha) = N(w|m_0, S_0)
        :param n_dim:
        :return:
        """
        self.m_0 = np.zeros(n_dim)
        self.S_0 = 1 / self.alpha * np.eye(n_dim)

    def fit(self, X, y):
        if (X.ndim == 1):
            D = 1
            X = X[:, None]
        else:
            D = X.shape[1]
        self.init_params(D)

        self.SN = np.linalg.inv(np.linalg.inv(self.S_0) + self.beta * (X.T @ X))
        self.mN = self.SN @ (np.linalg.inv(self.S_0) @ self.m_0 + self.beta * X.T @ y)


    def draw_posterior(self):
        """
        posterior likelihood p(w|y,X) = N(w|mN, SN)
        :param ax:
        :return:
        """
        w0, w1 = np.meshgrid(np.linspace(-5, 5, 100),
                        np.linspace(-5, 5, 100))
        w = np.array([w0, w1]).transpose(1, 2, 0)
        plt.contourf(w0, w1, multivariate_normal.pdf(w, mean=self.mN, cov=self.SN), cmap="rainbow")

    def draw_likelihood(self, X, y):
        w0, w1 = np.meshgrid(np.linspace(-1, 1, 100),
                        np.linspace(-1, 1, 100))
        w = np.array([w0, w1]).transpose(1, 2, 0)
        plt.contourf(w0, w1, self.log_likelihood(w, X, y), cmap="rainbow")

    def predict(self, X, n_samples=5):
        """
        return means and variances of predictive distribution
        :param X:  (N, D)
        :param n_samples:
        :return:
        """
        # w.shape = (n_samples, D)
        w = np.random.multivariate_normal(mean=self.mN, cov=self.SN, size=n_samples)
        y_pred = X @ w.T
        y_std = np.diagonal(1/self.beta + X@self.SN@(X.T))
        return y_pred, y_std

    def log_likelihood(self, w, X, y):
        """
        return log-likelihood of p(y|X,w)
        :param w: (100, 100, 2)
        :param X: (D, )
        :param y: (1, )
        :return:
        """
        ndim = w.shape[-1]
        Nw = w.shape[0]
        weights = w.reshape((-1, ndim))
        N = weights.shape[0]

        P = []
        for i in range(N):
            P.append(np.sqrt(self.beta) \
                     / np.sqrt(2*np.pi) \
                     * np.exp((-0.5 * self.beta
                               * np.sum(np.square(y - weights[i] @ X.T)))))

        P = np.array(P).reshape(Nw, Nw)
        return P