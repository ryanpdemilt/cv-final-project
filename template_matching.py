from constants import IM_DIM
import numpy as np
import skimage
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import os

WINDOW_FIELD = 30

def match_image(frame, letter):

    # for now template img will be first image in that letters folder
    template_img = plt.imread(os.path.join(letter, f'{letter}_template.png'))

    flattened_template = template_img.reshape(-1,template_img.shape[-1])
    t_mean = np.mean(flattened_template,axis=0)
    t_std = np.std(flattened_template,axis=0)

    window_size = template_img.shape
    windows = skimage.util.view_as_windows(frame,window_size)
    error_surface = np.zeros(windows.shape[0:2])
    for i in range(int(windows.shape[0] / 2) - WINDOW_FIELD,int(windows.shape[0] / 2) + WINDOW_FIELD,1):
        for j in range(int(windows.shape[1] / 2) - WINDOW_FIELD,int(windows.shape[1] / 2) + WINDOW_FIELD,1):
            patch = windows[i,j,0]
            flattened_patch = patch.reshape(-1, patch.shape[-1])

            p_mean = np.mean(flattened_patch,axis=0)
            p_std = np.std(flattened_patch,axis=0)

            error_surface[i,j] =  np.sum(((flattened_patch - p_mean) * (flattened_template - t_mean))/ (p_std*t_std))
            
    error_surface = error_surface * (1 / (flattened_template.shape[0] - 1))

    ind = np.unravel_index(np.argmax(error_surface, axis=None), error_surface.shape)
    print(ind)
    plt.imshow(windows[ind[0],ind[1],0])
    plt.show()

    print(error_surface[ind[0],ind[1]])


frame = plt.imread(os.path.join('A','A_45.png'))
match_image(frame, 'A')

frame = plt.imread(os.path.join('S','S_80.png'))
match_image(frame, 'A')