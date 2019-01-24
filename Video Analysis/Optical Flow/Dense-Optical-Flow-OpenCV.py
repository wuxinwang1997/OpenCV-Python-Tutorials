# -*- coding: utf-8 -*-
"""
@author: wuxin
"""

import cv2 as cv
import numpy as np

if __name__ == '__main__':

    cap = cv.VideoCapture('vtest.avi')

    ret, frame1 = cap.read()

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('Dense Optical Flow in OpenCV.avi', fourcc, 20.0, (frame1.shape[1], frame1.shape[0]))

    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    while(1):
        ret, frame2 = cap.read()
        if ret == True:
            next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

            flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 1, 5, 1.2, 0)

            mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
            bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

            out.write(bgr)
            cv.imshow('Original', frame2)
            cv.imshow('Dense Optical Flow in OpenCV', bgr)
            k = cv.waitKey(60) & 0xff
            if k == 27:
                break

            prvs = next
        else:
            break

    out.release()
    cap.release()
    cv.destroyAllWindows()
