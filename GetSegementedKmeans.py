import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def getSegmentedKMeans(img):
  img_array = img.reshape(154401,3)
  k=np.array([3,5,7,9,11])

  # 5 ->number of Ks
  # 481 * 321 -> image size
  segmented_results = np.empty([5,481,321])
  for k,i in zip((k),range(5)):
    seg = (KMeans(n_clusters=k, random_state=0).fit(img_array))
    segmented_results[i] = np.array([j for j in seg.labels_]).reshape(481,321)
    plt.title("K means segmentation when K = "+str(k))
    plt.imshow(segmented_results[i])
    plt.show()
  return (segmented_results)