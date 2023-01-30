import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree

# Load the data
iris = load_iris()
x = iris.data[:, :2]
y = iris.target

# Train the model
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(x, y)

# Predict the values
y_pred = clf.predict(x)

# Plot the data and predictions
plt.scatter(x[:, 0], x[:, 1], c=y_pred)
plt.show()
