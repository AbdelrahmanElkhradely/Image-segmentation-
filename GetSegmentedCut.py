import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering, spectral_clustering
from sklearn.utils import graph


def getSegmentedNcut(img,Kvalues):
  img_array = img.reshape(154401, 3)
  segmented_results = np.empty([len(Kvalues), 481, 321])
  for k, i in zip((Kvalues), range(len(Kvalues))):
    seg =SpectralClustering(n_clusters=k, n_neighbors=5, assign_labels='kmeans',).fit(img_array)
    segmented_results[i] = np.array([j for j in seg.labels_]).reshape(481, 321)
    plt.imshow(segmented_results[i])
    plt.show()
  return (segmented_results)