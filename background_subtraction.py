import numpy as np
import os
import glob
import scipy
import cv2

def load_background_video():
    filelist = glob.glob(os.path.join('background','background_*.png'))
    background = np.array([np.array(cv2.imread(fname)) for fname in filelist])
    mean = np.mean(x,axis=0)
    std = np.std(x,axis=0)

    return background, mean, std

def background_subtract(sample,background_mean,background_std,threshold=150):
    background_subtracted_image = (np.square(sample - mean)/ np.square(std)) > 150
    return background_subtracted_image
