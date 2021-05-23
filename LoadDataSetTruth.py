import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import scipy.io as sio
from PIL import Image


def get_dataset_truth(base):
    BASE = './BSR/BSDS500/data/' + base + "/"
    filepaths = []
    for s_i in np.os.listdir(BASE):
        if s_i != 'README':
            filepaths.append(BASE + s_i)
    df = pd.DataFrame({'filepaths': filepaths})
    groundtruthimages  = []
    for filepath in df['filepaths']:
        groundtruthimages.append(filepath)

    # img2.save("your_file.jpeg")
    # print(temp)

    return groundtruthimages