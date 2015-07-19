from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
pca_2c = PCA(n_components=2)
X_pca_2c = pca_2c.fit_transform(iris.data)
print X_pca_2c.shape
plt.scatter(X_pca_2c[:, 0], X_pca_2c[:, 1], c=iris.target,
            alpha=0.8, edgecolors='none')
plt.show()
print pca_2c.explained_variance_ratio_.sum()

print X_pca_2c
