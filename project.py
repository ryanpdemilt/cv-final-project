import numpy as np
import scipy
import cv2
from constants import IM_DIM,BACKGROUND_FRAMES

import matplotlib.pyplot as plt
### Background Setup ###
# 1. Capture Background
# 2. Compute Statistics

# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     raise IOError("Cannot Open webcam")

# def background_load():
#     frame_count = 0
#     background = np.zeros((BACKGROUND_FRAMES,IM_DIM,IM_DIM))
#     for i in range(BACKGROUND_FRAMES):
#         ret, frame = cap.read()
#         frame = cv2.resize(frame, (IM_DIM,IM_DIM), interpolation=cv2.INTER_AREA)
#         background[i] = frame
#     return background

# background = background_load()
### Template Setup ###
# 1. Load sample images
# 2. Perform background subtraction


### Main Loop ###
# Capture camera video and compute classification

# Store last 5? seconds of video, compare to templates, output matching letter
print('about to start')
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
print('got the camera you bitch')

if not cap.isOpened():
    raise IOError("Cannot Open webcam")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow('Input', frame)
    c = cv2.waitKey(1)
    if c == 27:
        break
print('we left the loop')
cap.release()
cv2.destroyAllWindows()