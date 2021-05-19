import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2
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
    return groundtruthimages