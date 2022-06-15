import os

import numpy as np
import matplotlib.pyplot as plt
os.chdir("..")

from utils import ( generate_1D_samples,
                    linear_reg_1D,
                    sin_1D)
from BayesianLinear import BayesianLinear
from kernel.PolynomialBasisFunction import PolynomialBasisFunction
from kernel.GaussianBasisFunction import GaussianBasisFunction
from scipy.stats import multivariate_normal

Poly = PolynomialBasisFunction(degree=1)
w = [-0.3, 0.5]
# x_train, y_train = generate_1D_samples(linear_reg_1D,
#                                        n_samples=20,
#                                        w=w,
#                                        low=-1,
#                                        high=1,
#                                        error_std=0.2)
# X_train = Poly.transform(x_train)
# alpha = 2.0
# beta = 25
# bayes = BayesianLinear(alpha, beta)

########################################################
#
#                   sin(x)
#
########################################################
w = [1, 2*np.pi]
x_train, y_train = generate_1D_samples(sin_1D,
                                       n_samples=20,
                                       w = w,
                                       low = 0,
                                       high = 1,
                                       error_std=0.2)
x_test = np.linspace(0, 1, 100)
'''
plt.scatter(x_train, y_train,
            s=100, facecolor="none",
            edgecolors="steelblue")
plt.show()
'''

GBF = GaussianBasisFunction(np.linspace(-1, 1, 10), 10)
X_train = GBF.transform(x_train)
X_test = GBF.transform(x_test)

model = BayesianLinear(alpha=1e-3, beta=2.)

model.fit(X_train[0:20], y_train[0:20])
y_pred, y_std = model.predict(X_test)
plt.scatter(x_train[0:20], y_train[0:20],
            s=50, facecolor="none",
            edgecolors="steelblue")
y_mean = np.mean(y_pred, axis=1)
plt.plot(x_test, y_mean, color="blue")

plt.fill_between(x_test, y_mean-y_std, y_mean+y_std,
                 color="orange", alpha=0.5)
plt.show()
