import pandas as pd
import matplotlib.pyplot as plt


iris_file_path = '../data/datasets-uci-iris.csv'
iris = pd.read_csv(iris_file_path, header=None,
                   names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'])
print iris.head()
print iris.describe()
print iris.quantile([0.1, 0.25, .5, .75, .9])
print iris.target.unique()

# Co-occurrence Matrix
print pd.crosstab(iris['petal_length'] > iris['petal_length'].mean(), iris['petal_width'] > iris['petal_width'].mean())
plt.scatter(iris['petal_width'], iris['petal_length'], alpha=1.0, color='k')
plt.xlabel("petal width")
plt.ylabel("petal length")
plt.show()

plt.hist(iris['petal_width'], bins=20)
plt.xlabel("petal width distribution")
plt.show()


# Box Plot
box_plot = iris.boxplot()
plt.show()

