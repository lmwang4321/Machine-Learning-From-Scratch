import numpy as np
from numpy import linalg as la
from scipy.special import gamma

def wishart(Lambda, W, v):
    """
    wishart distribution:

    wishart(Lambda | W, v) = B*det(Lambda)^((v-D-1)/2)*exp{-1/2 Tr(inv(W)*Lambda}
    B(W, v) = det(W)^(-v/2)*(2*(vD/2)*(pi^(D(D-1))/4)

    :param Lambda:  random variable, in this case
                    Lambda is the covariance matrix
                    for mu. D x D matrix
    :param W:       hyperparameter: D x D positive definite matrix
    :param v:       hyperparameter: real
    :return:
    """
    # Dimension
    D = W.shape[0]
    # Calculate B(W, v)
    # |W| is real
    W_half = np.pow(la.det(W), 2)
    multi_gamma = 0
    for i in range(D):
        multi_gamma *= gamma((v + 1 - i) / 2)
    B = W_half * np.pow((np.pow(2, v * D / 2)) \
                        * np.pow(np.pi, D * (D - 1) / 4) * multi_gamma, -1)
    return B * np.pow(la.det(Lambda), (v - D - 1) / 2) \
           * np.exp(-1 / 2 * np.trace(la.inv(W) @ Lambda))