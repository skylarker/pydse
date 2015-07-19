import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
cov_data = np.corrcoef(iris.data.T)
print iris.feature_names
print cov_data
img = plt.matshow(cov_data, cmap=plt.cm.winter)
plt.colorbar(img, ticks=[-1, 0, 1])
plt.show()