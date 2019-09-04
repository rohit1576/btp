import copy
import cv2
import numpy as np
from phue import Bridge
from soco import SoCo
import pygame
import time

gesture_names = {0: 'Fist',
                 1: 'L',
                 2: 'Okay',
                 3: 'Palm',
                 4: 'Peace'}

cap_region_x_begin = 0.5  # start point/total width
cap_region_y_end = 0.8  # start point/total width
threshold = 60  # binary threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

# variableslt
isBgCaptured = 0  # bool, whether the background captured
triggerSwitch = False  # if true, keyboard simulator works

def remove_background(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

    return result, score

camera = cv2.VideoCapture(0) #uses webcam for video

bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)

i = 0
capture = False

while camera.isOpened():
    #ret returns True if camera is running, frame grabs each frame of the video feed
    
    ret, frame = camera.read()
    # cv2.imshow("input", frame)
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)

    cv2.imshow('original', frame)

    k = cv2.waitKey(10)

    
        
    img = remove_background(frame)
    img = img[0:int(cap_region_y_end * frame.shape[0]),
            int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
    # cv2.imshow('mask', img)

    # convert the image into binary image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
    # cv2.imshow('blur', blur)
    ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imshow('ori', thresh)

    # img_name = "./dataset/peace/peace_{}.png".format(i)
    # cv2.imwrite(img_name, thresh)
    # print("{} written!".format(img_name))

    i += 1
    # time.sleep(1)

camera.release()
cv2.destroyAllWindows()
