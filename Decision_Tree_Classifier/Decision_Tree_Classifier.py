from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn.tree import plot_tree

canser = load_breast_cancer()
feature_names = canser.feature_names
target_names = canser.target_names

x_train, x_test, y_train, y_test = train_test_split(canser.data, canser.target, random_state = 42)
Tree = DecisionTreeClassifier(random_state = 0)
Tree.fit(x_train, y_train)

plt.figure(figsize = (20, 10))
plot_tree(Tree, filled = True)
plt.show()