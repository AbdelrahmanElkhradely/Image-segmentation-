import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import matplotlib.image as mpimg
from PIL import Image
import  GetSegementedKmeans
def visualize(images,matrix):
    imagescount=0
    for m in matrix:
        mat = scipy.io.loadmat(m)
        pil_im = Image.open(images[imagescount])
        img_array = np.asarray(pil_im)
        plt.imshow(img_array)
        plt.show()
        fig = plt.figure(figsize=(30, 50))
        for i in range((mat['groundTruth'].shape[1] )):
            seg = np.array(mat['groundTruth'][0, i][0, 0][0])
            bud = np.array(mat['groundTruth'][0, i][0, 0][1])
            # imagefig = plt.figure(figsize=(30, 50))
            # image=images[imagescount]
            # plt.show(image)
            fig.add_subplot((mat['groundTruth'].shape[1]), 2, (i * 2 + 1))
            plt.imshow(seg)
            fig.add_subplot((mat['groundTruth'].shape[1]), 2, (i * 2 + 1) + 1)
            plt.imshow(bud)
        plt.show()
        imagescount+=1
        GetSegementedKmeans.getSegmentedKMeans(img_array)

