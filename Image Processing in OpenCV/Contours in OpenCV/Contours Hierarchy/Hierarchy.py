# -*- dofing: utf-8 -*-
"""
@ author: wuxin
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":

    img = cv.imread('Layers.png')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret1, thresh = cv.threshold(gray, 127, 255, 0)

    image, contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, 1)

    print(hierarchy)

    cv.imshow('image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()

