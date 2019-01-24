#-*- coding:utf-8 -*-
"""
@ author: wuxin
"""

import cv2 as cv
import numpy as np

if __name__ == "__main__":

    img = cv.imread('opencv.jpg', 0)
    img = cv.medianBlur(img, 5)
    cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 20,
                              param1=60, param2=32, minRadius=0, maxRadius=0)

    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv.imshow('detected circles', cimg)
    cv.waitKey(0)
    cv.destroyAllWindows()
