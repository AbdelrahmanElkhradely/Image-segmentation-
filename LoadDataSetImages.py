import numpy as np
import pandas as pd

import cv2
def get_dataset_images(base):
    BASE = './BSR/BSDS500/data/'+base+"/"
    filepaths = []
    for s_i in np.os.listdir(BASE):
        if s_i != 'README':
                filepaths.append(BASE + s_i  )
    df = pd.DataFrame({'filepaths':filepaths})
    images = []
    filepathofimages=[]
    for filepath in df['filepaths']:
        images.append(cv2.imread(filepath, 0).flatten())
        filepathofimages.append(filepath)
    return images,filepathofimages


