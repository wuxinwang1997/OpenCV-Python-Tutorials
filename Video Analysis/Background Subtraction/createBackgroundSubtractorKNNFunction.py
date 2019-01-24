# -*- coding: utf-8 -*-
"""
@author: wuxin
"""

import cv2 as cv
import numpy as np

if __name__ == '__main__':
    cap = cv.VideoCapture('vtest.avi')
    ret1, frame1 = cap.read()
    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('Background Subtraction KNN.avi', fourcc, 20.0, (frame1.shape[1], frame1.shape[0]))

    fgbg = cv.createBackgroundSubtractorKNN()

    while(1):
        ret, frame = cap.read()

        if ret == True:
            fgmask = fgbg.apply(frame)

            out.write(fgmask)
            cv.imshow('Background Subtraction KNN', fgmask)
            k = cv.waitKey(120) & 0xff
            if k == 27:
                break

        else:
            break

    out.release()
    cap.release()
    cv.destroyAllWindows()