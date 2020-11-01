# This algorithm is known a K nearest neighbours.
# @description: A sample is classified by a popularity vote by its neighbours.

# to calculate the distance between neighbours we use Euclidean Distance
import numpy as np
from collections import Counter
from euclidea_distance import euclidean_distance

class KNN:
  
  # K, are the number of nearest neighbours to consider
  def __init__(self, k=3):
    self.k = k

  def fit(self, X, y):
    self.X_train = X
    self.y_train = y

  def predict(self, X):
    predicted_labels = [self._predict(x) for x in X]
    return np.array(predicted_labels)

  def _predict(self, x):
    # compute the distances (Euclidean distance)
    distanes = [euclidean_distance(x, x_train) for x_train in self.X_train]

    # Get the K-nearest neighbours(samples)
    # we sort our distances
    k_indices = np.argsort(distanes)[: self.k]

    # labels of the samples
    k_nearest_labels = [self.y_train[i] for i in k_indices]

    # do a maturity vote-get the most common class label
    most_common = Counter(k_nearest_labels).most_common(1)
    return most_common[0][0]