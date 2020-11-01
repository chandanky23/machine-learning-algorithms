# This alogorithm is used to predict continues data.
# To solve this algorithm we will try and minimise our cost function using the technique called Gradient Descent.

import numpy as np

class LinearRegression:

  def __init__(self, lr=0.01, n_iters=1000):
    self.lr = lr
    self.n_iters = n_iters
    self.weigths = None
    self.bias = None

  def fit(self, X, y):
    # init our parameters
    n_samples, n_features = X.shape
    self.weigths = np.zeros(n_features)
    self.bias = 0

    for _ in range(self.n_iters):
      y_approximate = np.dot(X, self.weigths) + self.bias
      
      dw = (1/n_samples) * np.dot(X.T, (y_approximate - y))
      db = (1/n_samples) * np.sum(y_approximate-y)

      self.weigths -= self.lr * dw
      self.bias -= self.lr * db

  def predict(self, X):
    y_predicted = np.dot(X, self.weigths) + self.bias
    return y_predicted