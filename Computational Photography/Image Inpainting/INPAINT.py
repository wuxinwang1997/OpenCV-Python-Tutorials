# -*- coding: utf-8 -*-
"""
@ author: wuxin
"""

import cv2 as cv
import numpy as np

if __name__ == '__main__':

    img = cv.imread('messi2.jpg')
    mask = cv.imread('mask2.png', 0)

    dst1 = cv.inpaint(img, mask, 3, cv.INPAINT_TELEA)
    dst2 = cv.inpaint(img, mask, 3, cv.INPAINT_NS)

    cv.imshow('dst1', dst1)
    cv.imshow('dst2', dst2)
    cv.waitKey(0)
    cv.destroyWindow()