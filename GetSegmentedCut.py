import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering, spectral_clustering
from sklearn.utils import graph
from PIL import Image
import cv2


def getSegmentedNcut(img,Kvalues):
  img = cv2.imread(img)
  img=cv2.resize(img,(0,0),fx=0.15,fy=0.15)
  d=img.shape
  print(d)
  # segmented_results = np.empty([len(Kvalues), 481, 321])
  segmented_results = np.zeros([len(Kvalues), d[0], d[1]])
  img=img.reshape( (-1, 3) )
  img=np.float32(img)
  seg =SpectralClustering(n_clusters=5,affinity='nearest_neighbors',random_state=0, n_neighbors=5,).fit(img)
    # segmented_results[i] = np.array([j for j in seg.labels_]).reshape(481, 321)
  segmented_results[0] = np.array([j for j in seg.labels_]).reshape( d[0], d[1])
  plt.imshow(segmented_results[0])
  plt.show()
  return (segmented_results)