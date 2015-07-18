from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


iris = datasets.load_iris()
print iris.DESCR
print iris.data
print iris.data.shape
print iris.feature_names
print iris.target
print iris.target.shape
print iris.target_names

colors = list()
palette = {0: 'red', 1: 'green', 2: 'blue'}

for c in np.nditer(iris.target):
    colors.append(palette[int(c)])

dataframe = pd.DataFrame(iris.data, columns=iris.feature_names)
scatter_plot = pd.scatter_matrix(dataframe, alpha=0.3, figsize=(10, 10), diagonal='hist', color=colors, marker='o',
                                 grid=True)
print scatter_plot
plt.show()