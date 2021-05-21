import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from PIL import Image
import  GetSegementedKmeans
import GetSegmentedCut
import getFmeasure
import getConditionalEntropy

def visualize(images,matrix):
        imagescount=0
        m=matrix
        print(m)
        mat = scipy.io.loadmat(m)
        # pil_im = Image.open(images[imagescount])
        pil_im=Image.open(images)
        img_array = np.asarray(pil_im)
        plt.imshow(img_array)
        plt.show()
        fig = plt.figure(figsize=(30, 50))
        for i in range((mat['groundTruth'].shape[1] )):
            seg = np.array(mat['groundTruth'][0, i][0, 0][0])
            bud = np.array(mat['groundTruth'][0, i][0, 0][1])
            fig.add_subplot((mat['groundTruth'].shape[1]), 2, (i * 2 + 1))
            plt.imshow(seg)
            fig.add_subplot((mat['groundTruth'].shape[1]), 2, (i * 2 + 1) + 1)
            plt.imshow(bud)
        plt.show()
        imagescount+=1
        # segmented_reseult=GetSegementedKmeans.getSegmentedKMeans(img_array,np.array([3,5,7,9,11]))
        # getFmeasure.getFMeasure(segmented_reseult,mat)
        # getConditionalEntropy.getConditionalEntropy(segmented_reseult,mat)
        print("###############################################################################")
        print("###############################################################################")
        GetSegmentedCut.getSegmentedNcut(img_array,np.array([5]))

