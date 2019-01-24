# -*- dofing: utf-8 -*-
"""
@ author: wuxin
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":

    img = cv.imread('boxing.jpg')

    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 127, 255, 0)
    image, contours1, hierarchy1 = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    IMAGE, contours2, hierarchy2 = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    img = cv.drawContours(image, contours1, -1, (0, 255, 0), 3)
    dst = cv.drawContours(IMAGE, contours2, 0, (0, 255, 0), 3)

    cv.imshow("Original", img)
    cv.imshow("Average", dst)
    cv.waitKey(0)
    cv.destroyAllWindows()