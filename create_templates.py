import numpy as np
import scipy
import cv2
import os
import matplotlib.pyplot as plt

from constants import IM_DIM, BACKGROUND_FRAMES

frame_count = 100

def get_letter(letter, cap):
    if not os.path.exists(os.path.join(letter, 'test')):
        os.mkdir(os.path.join(letter, 'test'))

    for i in range(frame_count):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (IM_DIM, IM_DIM), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imshow('Input', frame)    
        c = cv2.waitKey(1)
        
        if c == 27:
            break

        cv2.imwrite(os.path.join(letter,'test',f'{letter}_{i}.png'), frame)

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


letters = ['S']
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

for letter in letters:
    lets_wait = input(f'Now recording letter {letter}')
    get_letter(letter, cap)

cap.release()
cv2.destroyAllWindows()