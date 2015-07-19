import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


N_samples = 2000
dataset_1 = np.array(datasets.make_circles(n_samples=N_samples,
                                           noise=0.05, factor=0.3)[0])
dataset_2 = np.array(datasets.make_blobs(n_samples=N_samples, centers=4, cluster_std=0.4, random_state=0)[0])
plt.scatter(dataset_1[:, 0], dataset_1[:, 1], c='k', alpha=0.8, s=5.0)
plt.show()
plt.scatter(dataset_2[:, 0], dataset_2[:, 1], c='k', alpha=0.8, s=5.0)
plt.show()

K_dataset_1 = 2
km_1 = KMeans(n_clusters=K_dataset_1)
labels_1 = km_1.fit(dataset_1).labels_
plt.scatter(dataset_1[:, 0], dataset_1[:, 1], c=labels_1,
            alpha=0.8, s=5.0, lw=0)
plt.scatter(km_1.cluster_centers_[:, 0], km_1.cluster_centers_[:, 1],
            s=100, c=np.unique(labels_1), lw=0.2)
plt.show()

K_dataset_2 = 4
km_2 = KMeans(n_clusters=K_dataset_2)
labels_2 = km_2.fit(dataset_2).labels_
plt.scatter(dataset_2[:, 0], dataset_2[:, 1], c=labels_2,
            alpha=0.8, s=5.0, lw=0)
plt.scatter(km_2.cluster_centers_[:, 0], km_2.cluster_centers_[:, 1],
            s=100, c=np.unique(labels_2), lw=0.2)
plt.show()



