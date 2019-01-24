# -*- coding: utf-8 -*-
"""
@author: wuxin
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":

    img = cv.imread('opencv.jpg')

    blur = cv.GaussianBlur(img, (5, 5), 0)

    plt.subplot(121), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(blur), plt.title('GaussianBlur')
    plt.xticks([]), plt.yticks([])
    plt.show()