import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

from logistic_regression import LogisticRegression

bc = datasets.load_breast_cancer()

X, y = bc.data, bc.target

x_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

def accuracy(y_true, y_pred):
  accuracy = np.sum(y_true == y_pred) / len(y_true)
  return accuracy

regression = LogisticRegression(lr=0.01, n_iters=1000)
regression.fit(x_train, y_train)
predictions = regression.predict(X_test)

print("Logistic Regression accuracy: {}".format(accuracy(y_test, predictions)))