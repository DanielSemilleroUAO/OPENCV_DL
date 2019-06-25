#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 18:57:07 2019

@author: daniel
"""
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import cv2 as cv

cvNet = cv.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'graph.pbtxt')

#vs = VideoStream(src=1).start()
#frame = vs.read()
#cv2.destroyAllWindows()
#vs.stop()
#img = imutils.resize(frame, width=300)
#(h, w) = frame.shape[:2]
img = cv.imread('prueba1.jpg')
img = imutils.resize(img, width=300)
#img = imutils.rotate(img, 90)
img = imutils.rotate_bound(img, 180)
rows = img.shape[0]
cols = img.shape[1]
cvNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
cvOut = cvNet.forward()

for detection in cvOut[0,0,:,:]:
    score = float(detection[2])
    if score > 0.8:
        class_id = detection[1]
        print(class_id,score)
        left = detection[3] * cols
        top = detection[4] * rows
        right = detection[5] * cols
        bottom = detection[6] * rows
        cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)

cv.imshow('img', img)
cv.waitKey()
#plt.imshow(img)