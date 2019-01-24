# -*- coding:utf- -*-
"""
@author: wuxin
"""

import cv2 as cv
import numpy as np

if __name__ == '__main__':

    cap = cv.VideoCapture('car.mp4')

    # take first frame of the video
    ret, frame = cap.read()

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('Camshift.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))

    # setup initial location of window
    r, h, c, w = 320,30,150,30    # simply hardcoded the value
    track_window = (c, r, w, h)

    # set up the ROI for tracking
    roi = frame[r:r+h, c:c+w]
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

    # Setup the termination critieria, either 10 iteration or move by altest 1 pt
    term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

    while(1):
        ret, frame = cap.read()

        if ret == True:
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

            # apply  meanshift to get the new location
            ret, track_window = cv.CamShift(dst, track_window, term_crit)

            # Draw it on image
            x,y,w,h = track_window
            img2 = cv.rectangle(frame, (x,y), (x+w, y+h), 255, 2)
            out.write(img2)
            cv.imshow('Meanshift', img2)

            k = cv.waitKey(60) & 0xff
            if k == 27:
                break

        else:
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()
