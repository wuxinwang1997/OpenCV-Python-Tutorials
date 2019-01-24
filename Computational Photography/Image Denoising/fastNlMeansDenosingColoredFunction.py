# -*- coding:utf-8 -*-
"""
@ author: wuxin
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    img = cv.imread('ellipses.jpg')

    dst = cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    plt.subplot(121), plt.imshow(img)
    plt.subplot(122), plt.imshow(dst)
    plt.show()