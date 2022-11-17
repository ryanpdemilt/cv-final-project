import numpy as np
import scipy
import cv2

import matplotlib.pyplot as plt


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot Open webcam")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow('Input', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break
cap.release()
cv2.destroyAllWindows()