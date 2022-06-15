from utils import ( sin_1D,
                    generate_1D_samples)
import numpy as np
from kernel.SquaredExponential import SquaredExponential as SE
import matplotlib.pyplot as plt

w = [1, np.pi]
s = 1
N = 4         # number of existing observation points (training points).

X, y = generate_1D_samples( sin_1D,
                            w = w,
                            low = -1,
                            high = 1,
                            n_samples = N,
                            error_std = s)


f = lambda x: np.sin(0.9*x).flatten()

# Sample some input points and noisy versions of the function evaluated at
# these points.
n = 200        # number of test points.
s = 0.00005    # noise variance.

# X = np.random.uniform(-5, 5, size=(N,1))     # N training points
# y = f(X) + s*np.random.randn(N)

K = SE(X, X)
L = np.linalg.cholesky(K + s*np.eye(N))     # line 1

# points we're going to make predictions at.
Xtest = np.linspace(-2, 2, n).reshape(-1,1)

# compute the mean at our test points.
Lk = np.linalg.solve(L, SE(X, Xtest))   # k_star = kernel(X, Xtest), calculating v := l\k_star
mu = np.dot(Lk.T, np.linalg.solve(L, y))    # \alpha = np.linalg.solve(L, y)

# compute the variance at our test points.
K_ = SE(Xtest, Xtest)                  # k(x_star, x_star)
s2 = np.diag(K_) - np.sum(Lk**2, axis=0)
s = np.sqrt(s2)

# PLOTS:
plt.figure(1)
plt.clf()
plt.scatter(X, y,
            s=100, facecolor="none",
            edgecolors="steelblue")
# plt.plot(Xtest, f(Xtest), 'b-')
plt.gca().fill_between(Xtest.flat, mu-2*s, mu+2*s, color="orange",
                       alpha=0.5)
plt.plot(Xtest, mu, 'r', lw=2)
# plt.savefig('predictive.png', bbox_inches='tight', dpi=300)
plt.title('Mean predictions plus 2 st.deviations')
plt.show()
#plt.axis([-5, 5, -3, 3])
# K = SE(X, l=0.1, sigma=1)
# N = K.shape[0]
# L = np.linalg.cholesky(K + s*np.eye(N))
# Xtest = np.linspace(-1, 1, 100)
# Lk = np.linalg.solve(L, SE(X, Xtest))
# mu = np.dot(Lk.T, np.linalg.solve(L, y))
# K_ = SE(Xtest, Xtest)
# s2 = np.diag(K_) - np.sum(Lk ** 2, axis=0)
# s = np.sqrt(s2)
# plt.scatter(X, y,
#             s=100, facecolor="none",
#             edgecolors="steelblue")
# plt.plot(Xtest, mu, color="red")
# plt.fill_between(Xtest,
#                  mu - 2*s,
#                  mu + 2*s,
#                  alpha=0.5)
# plt.show()
