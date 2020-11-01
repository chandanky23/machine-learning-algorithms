# This algorithm is known a K nearest neighbours.
# @description: A sample is classified by a popularity vote by its neighbours.

# to calculate the distance between neighbours we use Euclidean Distance

class KNN:
  
  # K, are the number of nearest neighbours to consider
  def __init__(self, k=3):
    self.k = k

  def fit(self, X, y):
    pass

  def predict(self, X):
    pass