import pandas as pd
import matplotlib.pyplot as plt


iris_file_path = '../data/datasets-uci-iris.csv'
iris = pd.read_csv(iris_file_path, header=None,
                   names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'])
print iris.head()
print iris.describe()
box_plot = iris.boxplot()
plt.show()

