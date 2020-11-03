# This is a classification algorithm. Here the features are considered to be mutually independent.
# Here the probability of each feature is calculated for given y.
# It uses the Gaussian normal distribution.

import numpy from np

class NaiveBayes:

  def fit(self, X, y):
    # Number of samples and number of features
    n_samples, n_features = X.shape
    
    self._classes = np.unique(y) # array with just unique value
    n_classes = len(self._classes)

    # init mean, variance, priors
    self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
    self._variance = np.zeros((n_classes, n_features), dtype=np.float64)
    self._priors = np.zeros(n_classes, dtype=np.float64)

    for c in self._classes:
      X_c = X[c==y]
      self._mean[c,:] = X_c.mean(axis=0)
      self._var[c,:] = X_c.var(axis=0)
      self._priors[c] = X_c.shape[0] / float(n_samples)

  def predict(self, X):
    y_predict = [self._predict(x) for x in X]
    return y_pred
  
  def _predict(self, x):
    posteriors = []
    for index, c in enumerate(self._classes):
      prior = np.log(self._priors[index])
      class_conditional = np.sum(np.log(self.probability_density_function(index, x)))
      posterior = prior + class_conditional
      posteriors.append(posterior)

    return self._classes[np.argmax(posteriors)]

  def probability_density_function(self, class_index, x):
    mean = self._mean[class_index]
    var = self._var[class_index]

    numerator = np.exp(- (x-mean)**2 / (2 * var))
    denominator = np.sqrt(2 * np.pi * var)

    return numerator/denominator