import numpy as np
from scipy.special import gamma
from numpy import linalg as la

def student_t(X, mu, Lambda, v):
    """

    Student t distribution:
    st(x | mu, Lambda, v) = gamma(D/2 + v/2)/gamma(v/2) * |Lambda|^(1/2)/(pi*v)^D/2
    :param X:       random variable n_sample x D
    :param mu:      mean            1 x D
    :param Lambda:  covariance      D X D
    :param v:       hyperparameter  real
    :return:
    """
    n_sample, D = X.shape
    ret = gamma(D / 2 + v / 2) * np.power(np.linalg.det(Lambda), 0.5) * np.power(
        1 + np.sum((X - mu) @ Lambda * (X - mu), axis=1) / v, -1.0 * D / 2 - v / 2) / gamma(v / 2) / np.power(
        np.pi * v, D / 2)
    return ret