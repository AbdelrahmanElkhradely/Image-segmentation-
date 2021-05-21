import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def getSegmentedKMeans(img,Kvalues):
  img_array = img.reshape(154401,3)
  k=Kvalues
  # 5 ->number of Ks
  # 481 * 321 -> image size
  size=len(Kvalues)
  segmented_results = np.empty([size,481,321])
  fig = plt.figure(figsize=(30, 50))
  font = {'size': 30}
  for k,i in zip((k),range(5)):
    seg = (KMeans(n_clusters=k, random_state=0).fit(img_array))
    segmented_results[i] = np.array([j for j in seg.labels_]).reshape(481,321)
    fig.add_subplot(5, 1, (i + 1))
    plt.imshow(segmented_results[i])
    plt.rc('font', **font)
    plt.title("K means segmentation when K = "+str(k))
    # plt.imshow(segmented_results[i])
  plt.show()
  return (segmented_results)