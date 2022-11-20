import numpy as np
import os
import glob
import scipy
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFilter

def load_background_video(grayscale=False):
    filelist = glob.glob(os.path.join('background','background_*.png'))
    if grayscale:
        background = np.array([np.array(cv2.GaussianBlur(cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2GRAY), (5,5), 0)) for fname in filelist])
    else:
        background = np.array([np.array(cv2.GaussianBlur(cv2.imread(fname), (5,5), 0)) for fname in filelist])
    mean = np.mean(background, axis=0)
    std = np.std(background, axis=0)

    return background, mean, std

def background_subtract(sample,background_mean,background_std,threshold=20, grayscale=False):
    
    blur_sample = cv2.GaussianBlur(sample,(5,5),0)

    if grayscale:
        gray_sample = cv2.cvtColor(blur_sample, cv2.COLOR_BGR2GRAY)
        background_mask = np.sqrt(np.square(gray_sample - mean)/ np.square(std)) < threshold
        gray_sample[background_mask] = 0
        return gray_sample
    else:
        background_mask = np.sqrt(np.sum((np.square(blur_sample - mean)),axis=2)) < threshold
        sample[background_mask] = 0
        return sample


# background, mean, std = load_background_video(grayscale=False)

# im_a = cv2.imread(os.path.join('S',f'S_45.png'))
# cool_im = background_subtract(im_a, mean, std, threshold=80, grayscale=False)
# cv2.imshow('Input', cool_im)
# c = cv2.waitKey(0)