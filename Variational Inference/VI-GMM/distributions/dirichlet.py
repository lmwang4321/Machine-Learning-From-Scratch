import numpy as np
from scipy.special import gamma

def dirichlet(u, alpha):
    """

    Dirichlet distribution:

    Dir(mu|alpha) = 1/B(alpha) * \multi_multiply_{i=1}^K x_i^{alpha_i-1}
    :param u:       random variable (mu_1, ..., mu_K)
    :param alpha:   hyperparameter  (alpha_1, ..., alpha_K)
    :return:
    """
    # Calculate B(alpha)
    dir = 0
    K = u.shape[0]
    for i in range(K):
        dir *= np.pow(u[i], alpha[i] - 1)
    B = gamma(np.sum(alpha))
    for alp in alpha:
        B /= gamma(alp)
    return dir * B