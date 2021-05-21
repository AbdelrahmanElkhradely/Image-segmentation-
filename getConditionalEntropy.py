
import numpy as np
import math

def getConditionalEntropy(segmented_results,mat):
  cond_entropy_list = []
  for m in range(len(segmented_results)):
    pred = segmented_results[m].flatten()
    for n in range(mat['groundTruth'].shape[1]):
      print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")
      print("Ground truth image number = " + str(n+1))
      truth = mat['groundTruth'][0, n][0, 0][0].flatten()
      predicted_labels = np.unique(pred)
      predicted_indicies = np.array([ np.where(pred == i) for i in predicted_labels if np.where(pred == i)[0].size > 0] ,dtype=object)
      clusters = [0 for x in predicted_labels]
      for i in range(len(predicted_labels)):
        clusters[i] = np.array([truth[j] for j in predicted_indicies[i]])
      cond_entropy = np.zeros(len(clusters))
      for i,c in zip(clusters,range(len(clusters))):
        sum_ = np.array([(np.array(i) == j).sum() for j in np.unique(truth)])
        entropy_ = np.array([j / (1.0 * len(i[0])) for j in sum_])
        entropy_ = np.array([math.log(j) * (j) * -1.0 for j in entropy_ if j != 0])
        cond_entropy[c] = sum(entropy_) * len(i[0]) * 1.0 / len(pred)
      print("number of clusters = " + str(len(clusters)))
      cond_entropy_list.append(sum(cond_entropy))
      print("Conditional Entropy:",sum(cond_entropy))
  return np.sum(cond_entropy_list)