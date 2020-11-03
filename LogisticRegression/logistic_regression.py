# In this model we use the linear regression model
# and then use the sigmoid function on top of it to get a probability between 0 and 1.

import numpy as np

class LogisticRegression:

  def __init__(self, lr=0.01, n_iters=1000):
    self.lr=lr
    self.n_iters = n_iters
    self.weights = None
    self.bias = None

  def fit(self, X, y):
    # initialize the parameter
    n_samples, n_features = X.shape
    self.weights = np.zeros(n_features)
    self.bias = 0

    # Use the gradient descent to iteratively update our weights and bias
    for _ in range(self.n_iters):
      linear_model = np.dot(X, self.weights) + self.bias
      y_approximate = self.sigmoid(linear_model)

      dw = 1/n_samples * np.dot(X.T, (y_approximate - y))
      db = 1/n_samples * np.sum(y_approximate-y)

      self.weights -= self.lr * dw
      self.bias -= self.lr * db

  def predict(self, X):
    linear_model = np.dot(X, self.weights) + self.bias
    y_approximate = self.sigmoid(linear_model)

    return [1 if i > 0.5 else 0 for i in y_approximate]

  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))
