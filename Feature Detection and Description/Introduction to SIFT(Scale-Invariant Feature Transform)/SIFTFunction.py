# -*- coding : ut-8 -*-
"""
@ author: wuxin
"""

import cv2 as cv
import numpy as np

if __name__ == "__main__":
    img = cv.imread('home.png')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sift = cv.xfeatures2d.SIFT_create()
    kp = sift.detect(gray, None)

    cv.drawKeypoints(gray, kp, img)

    cv.imshow('SIFT', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
