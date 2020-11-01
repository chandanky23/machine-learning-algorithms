# The test script is written using the IRIS datasets.

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000', '#00FF00', "#0000FF"])

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# print(X_train.shape)
"""
(120, 4)
"""
# print(X_train[0])
"""
[5.1 2.5 3.  1.1]
"""

# print(y_train.shape)
"""
(120,)
"""
# print(y_train)
"""
[1 2 0 2 1 0 0 0 0 1 0 1 0 2 2 0 2 2 2 2 0 2 2 1 1 1 1 1 1 0 0 2 2 2 0 0 0
 2 1 2 2 1 0 2 0 2 0 1 1 0 1 0 2 2 2 1 0 0 2 1 1 0 1 2 1 1 1 0 0 0 1 1 0 2
 1 2 2 1 0 1 2 0 0 2 2 1 1 2 0 1 2 2 2 1 0 0 0 0 2 1 2 0 0 1 1 2 1 1 2 2 2
 0 2 0 0 2 2 1 0 0]
"""

# plt.figure()
# # plotting 2d case
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k', s=20)
# plt.show()

from KNN import KNN

clf = KNN(k=5) # use odd numbers of neighbours, k=3 for example

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

# lets calculate the accuracy (how many of our predictions are correctkly classified with respect to y_test)
acc = np.sum(predictions == y_test) / len(y_test)

print('accuracy={}'.format(acc))
"""
for K=5
accuracy=0.9666666666666667
"""
"""
for k=3
accuracy=1.0
"""