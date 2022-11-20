from constants import IM_DIM
import numpy as np
import skimage
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import os
from tqdm import tqdm

WINDOW_FIELD = 50

def match_image(frame, letter):

    template_img = plt.imread(os.path.join(letter, f'{letter}_template.png'))

    flattened_template = template_img.reshape(-1,template_img.shape[-1])
    t_mean = np.mean(flattened_template,axis=0)
    t_std = np.std(flattened_template,axis=0)

    window_size = template_img.shape
    windows = skimage.util.view_as_windows(frame,window_size)
    error_surface = np.zeros(windows.shape[0:2])
    
    x = np.array(range(int(windows.shape[0] / 2) - WINDOW_FIELD,int(windows.shape[0] / 2) + WINDOW_FIELD,1))
    y = np.array(range(int(windows.shape[1] / 2) - WINDOW_FIELD,int(windows.shape[1] / 2) + WINDOW_FIELD,1))
    #(112, 161, 1, 145, 96, 3)
    flattened_shape = (windows.shape[0],windows.shape[1],windows.shape[2],windows.shape[3]*windows.shape[4],windows.shape[5])
    flattened_windows = np.reshape(windows,flattened_shape)
    # print(windows.shape)
    # print(flattened_windows.shape)
    xv,yv = np.meshgrid(x,y)

    patches = flattened_windows[xv,yv,0]
    # print(patches.shape)
    patch_means = np.mean(patches,axis=2)
    # patch_means = patch_means.reshape(-1,patch_means.shape[-1])
    # print(patch_means.shape)
    patch_stds = np.std(patches,axis=2)
    # patch_stds = patch_stds.reshape([-1,patch_stds.shape[-1]])
    # print(patch_stds.shape)
   

    # print('creating error_surface')
    # # error_surface = ((patches - patch_means)* (flattened_template-t_mean)) / (patch_stds*t_std)
    # print(error_surface.shape)
    for i in range(60):
        for j in range(60):
            patch = patches[i,j]
            # flattened_patch = patch.reshape(-1, patch.shape[-1])
            # p_mean = np.mean(flattened_patch,axis=0)
            # p_std = np.std(flattened_patch,axis=0)
            p_mean = patch_means[i,j]
            p_std = patch_stds[i,j]

            error_surface[i,j] =  np.sum(((patch - p_mean) * (flattened_template - t_mean))/ (p_std*t_std))
            
    error_surface = error_surface * (1 / (flattened_template.shape[0] - 1))

    ind = np.unravel_index(np.argmax(error_surface, axis=None), error_surface.shape)
    # print(ind)
    # plt.imshow(windows[ind[0],ind[1],0])
    # plt.show()

    # print(error_surface[ind[0],ind[1]])
    return error_surface[ind[0],ind[1]]

def test_template_match():
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    accuracies = [0 for i in range(len(letters))]
    
    for letter_file in range(len(letters)):
        print(f'Doing letter {letters[letter_file]}')
        for i in tqdm(range(1,100)):
            num_match = 0
            for letter_template in letters:
                
                frame = plt.imread(os.path.join(letters[letter_file],f'{letters[letter_file]}_{i}.png'))
                match = match_image(frame, letter_template)

                if match > 2.5:
                    num_match = num_match + 1

        accuracies[letter_file] = num_match / 99

    for l,a in zip(letters, accuracies):
        print(f'Letter {l} identified with accuracy {a}')


test_template_match()