import numpy as np
from sklearn.metrics.cluster import contingency_matrix


def getFMeasure(segmented_results,mat):
  for i in range(len(segmented_results)):
    print("*****************************************")
    pred = segmented_results[i].flatten()
    f1_score_list = []
    for j in range(mat['groundTruth'].shape[1]):
      truth = mat['groundTruth'][0, j][0, 0][0].flatten()
      con = contingency_matrix(truth, pred, eps=None, sparse=False).T
      f1_score = 0.0
      for k in range(len(con)):
        avg_f1_score = 0
        index = np.argmax(con[k])
        # prec= number of same elements within cluster
        # recall= number of same elements within all elements
        prec = con[k][index] * 1.0 / sum(con[k])
        recall = con[k][index] * 1.0 / sum(con[:,index])
        f1_score += (2.0 * prec * recall / (prec + recall))
      f1_score /= len(con) * 1.0
      f1_score_list.append(f1_score)
      avg_f1_score+=f1_score
      print("F1-Score when Ground Truth [",j+1,"] Segment # [",i,"]: ",np.around(f1_score,decimals=2))
    print("Average= " +str(avg_f1_score))
  return