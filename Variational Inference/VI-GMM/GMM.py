from scipy import special
from scipy.stats import multivariate_normal as mvn
from numpy import linalg as la
import numpy as np
from scipy.special import digamma
from scipy import special

import utils.utils
from distributions.student_t import student_t
from utils.utils import generate_PSD_mat
from utils.utils import plot_contourf
from utils import utils as utils

class GMM:
    """
    Implementation of GMM using Variational Inference
    """

    def __init__(self,
                 n_components=None,
                 eps=None,
                 wishart_W_0=None,
                 wishart_v_0=None,
                 gaussian_mean_0=None,
                 gaussian_beta_0=None,
                 dirichlet_alpha_0=None,
                 max_iter=100):
        """

        Constructor

        :param n_components:      number of Gaussian Mixture Models
        :param eps:               convergence criterion
        :param wishart_W_0:       prior W for Wishart distribution   Wishart(Lambda | W, v)
        :param wishart_v_0:       prior v for Wishart distribution
        :param gaussian_mean_0:   prior mean for gaussian distribution N(mu | m, (beta*Lambda)^-1)
        :param gaussian_beta_0:   prior beta for gaussian distribution
        :param dirichlet_alpha_0: prior alpha for dirichlet distribution Dir(pi | alpha)
        :param max_iter:          max iteration
        """


        if n_components is None:
            self.n_components = 1
        else:
            self.n_components = n_components

        if eps is None:
            self.eps = 1e-6

        # prior parameters
        self.wishart_W_0 = wishart_W_0  # D x D
        self.wishart_v_0 = wishart_v_0  # real
        self.gaussian_mean_0 = gaussian_mean_0 # real
        self.gaussian_beta_0 = gaussian_beta_0 # real
        self.dirichlet_alpha_0 = dirichlet_alpha_0 # real

        self.max_iter = max_iter

        # variational parameters which will be updated
        # during iteration
        self.wishart_W = None  # (K, D, D)
        self.wishart_v = None  # (K, )
        self.gaussian_mean = None # (K, D)
        self.gaussian_beta = None # (K, )
        self.dirichlet_alpha = None # (K, )

        # responsibilities
        self.r = None # (n_samples, K)
        self.rho = None # (n_samples, K)

        self.N = np.empty((self.n_components, )) # (K, )
        self.x_bar = None # (K, D)
        self.S = None # (K, D, D)


    def init_params(self, X):
        """

        initialize prior parameters and variational parameters

        :return:
        """


        D = X.shape[1]
        # initialize wishat_W
        if self.wishart_W_0 is None:
            self.wishart_W_0 = np.identity(D)
        self.wishart_W = np.asarray([generate_PSD_mat(D)] * self.n_components)

        # initialize wishart_v
        if self.wishart_v_0 is None:
            self.wishart_v_0 = D
        self.wishart_v = np.asarray([self.wishart_v_0 + np.random.random()] * self.n_components) * 0.1

        # initialize gaussian_mean
        if self.gaussian_mean_0 is None:
            self.gaussian_mean_0 = 0.0
        self.gaussian_mean = np.random.random((self.n_components, D))*0.1

        # initialize gaussian_beta
        if self.gaussian_beta_0 is None:
            self.gaussian_beta_0 = 1.0
        self.gaussian_beta = self.gaussian_beta_0 + np.random.random((self.n_components,)) * 0.1

        # initialize dirichlet alpha
        if self.dirichlet_alpha_0 is None:
            self.dirichlet_alpha_0 = 1.0
        self.dirichlet_alpha = self.dirichlet_alpha_0 + np.random.random((self.n_components,)) * 0.1


    def update_resp(self, X):
        """

        for each k in self.n_components:
            update N[k], x_bar[k], S[k]

        :param X: (n_samples, self.D)
        :return:
        """


        D = X.shape[1]
        self.S = np.zeros((self.n_components, D, D))
        self.N = np.sum(self.r, axis=0)
        Ninv_tile = np.tile(np.expand_dims(1/self.N, axis=-1), (1, D))
        self.x_bar = np.multiply(Ninv_tile, self.r.T@X)
        for i in range(self.n_components):
            tmp = (self.r[:,[i]]*(X - self.x_bar[i])).T@(self.r[:,[i]]*(X-self.x_bar[i]))
            self.S[i, :, :] = (1/self.N)[i] * tmp



    def update_variational_params(self):
        """

        for each k in self.n_components:
            B[k] = B_0 + N[k]
            m[k] = (1/B[k]) * (B_0 * m_0 + N[k] * x_bar[k])
            W[k]^(-1) = W_0^(-1) + N[k] * S[k] + (B_0 * N[k])/(B_0 + N[k])
                        * (x_bar[k] - m_0)(x_bar[k] - m_0).T

        :return:
        """

        for k in range(self.n_components):
            self.dirichlet_alpha[k] = self.dirichlet_alpha_0 + self.N[k]
            self.gaussian_beta[k] = self.gaussian_beta_0 + self.N[k]
            self.gaussian_mean[k] = (1/self.gaussian_beta[k]) \
                                    * (self.gaussian_beta_0 * self.gaussian_mean_0
                                    + self.N[k] * self.x_bar[k])

            W_k_inv = np.linalg.inv(self.wishart_W_0) + self.N[k] * self.S[k] + (self.gaussian_beta_0 * self.N[
                k]) / (self.gaussian_beta_0 + self.N[k]) * np.transpose(
                self.x_bar[k] - self.gaussian_mean_0) @ (self.x_bar[k] - self.gaussian_mean_0)

            self.wishart_W[k] = la.inv(W_k_inv)
            self.wishart_v[k] = self.wishart_v_0 + self.N[k]



    def update_rho(self, X):
        """

        calculate rho_n_k

        """
        N = X.shape[0]
        D = X.shape[1]


        self.rho = np.zeros((N, self.n_components))
        for i in range(N):
            for j in range(self.n_components):
                # calculate E[ln|Lambda_k|]
                E_lambda = 0
                for x in range(1, D+1):
                    E_lambda += digamma((self.wishart_v[j] + 1 - x)/2)
                E_lambda += D * np.log(2) + np.log(la.det(self.wishart_W[j]))
                self.rho[i, j] += 0.5 * E_lambda
                # calculate E[ln(pi_k)]
                E_pi = 0
                E_pi += digamma(self.dirichlet_alpha[j]) \
                        - digamma(np.sum(self.dirichlet_alpha))

                self.rho[i, j] += E_pi
                # calculate E_{mu_k, Lambda_k}
                E_mu_lambda = 0
                # E_mu_lambda -= D/2 * np.log(2*np.pi)
                E_mu_lambda +=  (D/self.gaussian_beta[j]
                                + self.wishart_v[j] * (self.gaussian_mean[[j]] - X[[i]])
                                                        @ self.wishart_W[j]
                                                        @ ((self.gaussian_mean[[j]] - X[[i]]).T))
                self.rho[i, j] -= 0.5 * E_mu_lambda
                self.rho[i, j] -= D/2 * np.log(2 * np.pi)
                self.rho[i, j] = np.exp(self.rho[i, j])


    def update_r(self):
        """

        r[n,k] = rho[n,k] / sum(rho[n,:])

        :return:
        """
        N, K = self.rho.shape
        self.r = np.zeros((N, K))
        self.r = self.rho / np.sum(self.rho, axis=1, keepdims=True)

    def fit(self, X):
        """

        :param X:
        :return:
        """
        self.init_params(X)

        for _ in range(self.max_iter):
            self.update_rho(X)
            self.update_r()
            self.update_resp(X)
            self.update_variational_params()
            # self.get_predictive_prob(X)


    def get_predictive_prob(self, X):
        """
        caclulate P(x_hat | X)
        :return:
        """
        D = X.shape[1]
        alpha_hat = np.sum(self.dirichlet_alpha)
        P = 1 / alpha_hat
        SUM = 0
        for i in range(self.n_components):
            SUM += self.dirichlet_alpha[i] \
                   * student_t(X, self.gaussian_mean[i],
                               (self.wishart_v[i] + 1 - D) \
                               / (1 + self.gaussian_beta[i]) \
                               * self.wishart_W[i] * self.gaussian_beta[i],
                               self.wishart_v[i] + 1 - D)
        P *= SUM
        return P


gmm = GMM(n_components=5, max_iter=100)

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=500, centers=4, cluster_std=0.55)
X = X[:,::-1]
gmm.fit(X)
plot_contourf(X, gmm.get_predictive_prob, lines=10)
plt.show()