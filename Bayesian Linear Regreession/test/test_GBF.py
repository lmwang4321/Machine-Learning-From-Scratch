import os
os.chdir("..")

from kernel.GaussianBasisFunction import GaussianBasisFunction as GBF
import numpy as np
import matplotlib.pyplot as plt

def test_GBF():
    X = np.linspace(-1, 1, 100)
    mu = np.linspace(-1, 1, 10)
    gbf = GBF(mu, 0.1)
    phi = gbf.transform(X)
    plt.plot(X, phi)
    plt.show()

if __name__ == "__main__":
    test_GBF()
