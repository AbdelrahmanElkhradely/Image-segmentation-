import numpy as np
import matplotlib.pyplot as plt
import scipy.io

import cv2
def visualize(matrix):
    for m in matrix:
        mat = scipy.io.loadmat(m)
        mat = scipy.io.loadmat(m)
        fig = plt.figure(figsize=(30, 50))
        for i in range((mat['groundTruth'].shape[1] )):
            seg = np.array(mat['groundTruth'][0, i][0, 0][0])
            bud = np.array(mat['groundTruth'][0, i][0, 0][1])
            fig.add_subplot((mat['groundTruth'].shape[1]), 2, (i * 2 + 1))
            plt.imshow(seg)
            fig.add_subplot((mat['groundTruth'].shape[1]), 2, (i * 2 + 1) + 1)
            plt.imshow(bud)
        plt.show()

