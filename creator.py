import numpy as np
import cv2
import math
import operator
import time
import copy
# Function to find difference in frames


# parameters
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

# Reading frames at multiple instances from webcam to different variables


k = 1
time.sleep(5)
bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)

camera = cv2.VideoCapture(0)
camera.set(10, 200)


while camera.isOpened():

        ret, frame = camera.read()
        frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
        frame = cv2.flip(frame, 1)  # flip the frame horizontally
        cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                    (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)

        cv2.imshow('original', frame)
        
        # img = remove_background(frame)
        # img = img[0:int(cap_region_y_end * frame.shape[0]),
        #       int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
        # cv2.imshow('mask', img)

        # # convert the image into binary image
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        # cv2.imshow('blur', blur)
        # ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # cv2.imshow('ori', thresh)

        # # get the contours
        # thresh1 = copy.deepcopy(thresh)
        # contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # length = len(contours)
        # maxArea = -1
        # if length > 0:
        #     for i in range(length):  # find the biggest contour (according to area)
        #         temp = contours[i]
        #         area = cv2.contourArea(temp)
        #         if area > maxArea:
        #             maxArea = area
        #             ci = i

        #     res = contours[ci]
        #     hull = cv2.convexHull(res)
        #     drawing = np.zeros(img.shape, np.uint8)
        #     cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
        #     cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

        # cv2.imshow('output', drawing)

cv2.destroyAllWindows()
camera.release()
