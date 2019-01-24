# -*- coding: utf-8 -*-
"""
@ author: wuxin
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    imgL = cv.imread('left.jpg', 0)
    imgR = cv.imread('right.jpg', 0)

    stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(imgL, imgR)
    plt.imshow(disparity, cmap='gray')
    plt.show()
