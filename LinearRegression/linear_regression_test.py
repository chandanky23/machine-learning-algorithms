import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import datasets

import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=1000, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# fig = plt.figure(figsize=(8,6))
# plt.scatter(X[:, 0], y, color="b", marker="o", s=30)
# plt.show()

# print(X_train.shape)
"""
(800, 1)
"""
# print(y_train.shape)
"""
(800,)
"""

from linear_regression import LinearRegression

regression = LinearRegression(lr=0.01)
regression.fit(X_train, y_train)
predicted = regression.predict(X_test)

# calculate the mean squared error over the predicted value and the actual value ,i.e, y_test
def mse(y_true, y_predicted):
  return np.mean((y_true-y_predicted)**2)

mse_value = mse(y_test, predicted)
print(mse_value)
"""
  lr=0.01, n_iters = 1000
  383.2488464665803
"""

# Plot this data
y_pred_line = regression.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)

plt.plot(X, y_pred_line, color='black', linewidth=2, label="Prediction")
plt.show()