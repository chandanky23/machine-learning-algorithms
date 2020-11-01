# This alogorithm is used to predict continues data.
# To solve this algorithm we will try and minimise our cost function using the technique called Gradient Descent.

class LinearRegression:

  def __init__(self, lr=0.001, n_iters=1000):
    self.lr = lr
    self.n_iters = n_iters
    self.weigths = None
    self.bias = None

  def fit(self, X, y):
    pass

  def predict(self, X):
    pass