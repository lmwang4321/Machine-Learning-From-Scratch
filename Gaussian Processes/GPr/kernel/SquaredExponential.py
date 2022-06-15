import numpy as np

def SquaredExponential(*args, l=1, sigma=1):
    """
    f(x) = sigma^2 * exp{-(x1-x2)^2/2*(l^2)}
    :param x1:  (N, D)
    :param x2:  (N, D)
    :return:
    """
    x1 = args[0]
    x2 = x1

    if (len(args) > 1):
        x2 = args[1]

    if x1.ndim == 1:
        x1 = x1[:, None]

    if x2.ndim == 1:
        x2 = x2[:, None]

    x1 = np.expand_dims(x1, axis=1)
    x2 = np.expand_dims(x2, axis=1)

    diff = x1 - x2.transpose(1, 0, 2)
    f = sigma**2 * \
        np.exp(-np.sum(np.square(diff), axis=2) \
               /(2*l))
    return f


