import random
import math
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class Kmeans:
  def __init__(self, n_clusters):
    self.n_clusters = n_clusters
    self.k = [[0 for _ in range(n_clusters)] for _ in range(len(X))]
    self.mu = None


  def initialize_mu(self, X):
    return random.choices(X, k=self.n_clusters)

  def compute_distance(self, Xi, mu_k):
    return math.sqrt(sum((Xi-mu_k)**2))

  def estep(self, X):
    self.mu = self.initialize_mu(X)
    # print(len(self.mu))
    for j in range(len(X)):
      dist = self.compute_distance(X[j], self.mu[0])
      self.k[j][0] = 1
      for i in range(1, len(self.mu)):
        if (self.compute_distance(X[j], self.mu[i])) < dist:
          self.k[j][i] = 1
          dist = self.compute_distance(X[j], self.mu[i])
          # make all other k[j][z] = 0 where z != i
          for z in range(self.n_clusters):
            # print("z: ",z)
            if z != i:
              self.k[j][z] = 0
  def mstep(self, X):
    for i in range(self.n_clusters):
      numerator = [0]*len(X[0])
      denominator = 0
      for j in range(len(X)):
        for m in range(len(X[0])):
          numerator[m] += self.k[j][i]*X[j][m]
        denominator += self.k[j][i]
      for z in range(len(X[0])):
        self.mu[i][z] = numerator[z]/denominator

  def fit(self, X):
    for _ in range(1000):
      self.estep(X)
      self.mstep(X)
      # print(self.mu)
      return self.mu, self.k

X, _ = make_blobs(n_samples=1000, n_features=2, centers=4)
kmeans = Kmeans(n_clusters=4)
mu, k = kmeans.fit(X)
plt.scatter(X[:,0], X[:, 1])
for i in range(4):
  plt.scatter(mu[i][0], mu[i][1])

plt.show()