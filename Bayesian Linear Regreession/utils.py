import numpy as np
import matplotlib.pyplot as plt

def generate_1D_samples(func,
                        n_samples,
                        w,
                        low,
                        high,
                        error_std=2):
    """
    generate training samples
    :param func:        y_train = mvn(mean=func(xtrain), cov=error_std)
    :param n_samples:
    :param low:
    :param high:
    :param dim:
    :param error_std:
    :return: float
    """
    x_train = np.linspace(low, high, n_samples)
    y_train = func(x_train, w[0], w[1]) + np.random.normal(loc=0,
                                               scale=error_std,
                                               size=x_train.size)
    return x_train, y_train

def linear_reg_1D(x, w0, w1):
    """
    1D linear regression
    :param x:   random variable
    :param w0:  parameter
    :param w1:  parameter
    :return:
    """
    return w0 + w1 * x

def sin_1D(x, A=1, w=1):
    """

    :param x: random variable
    :param A: amplitude, float
    :param w: shift, float
    :return:
    """
    return A*np.sin(w * x)


def gaussian(x, mean=None, precision=None):
    """
    1D gaussian distribution
    f(x) = 1/sqrt(2pi)
    :param x:           random variable (N,)
    :param mean:        mean            float
    :param precision:   precision       float
    :return:
    """
    if mean is None:
        mean = 0
    if precision is None:
        precision = 1

    return 1/np.sqrt(2 * np.pi) * precision * np.exp(-0.5*(((x-mean)**2)*precision))

