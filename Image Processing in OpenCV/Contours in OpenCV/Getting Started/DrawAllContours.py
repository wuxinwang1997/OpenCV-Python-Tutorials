# -*- dofing: utf-8 -*-
"""
@ author: wuxin
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":

    im = cv.imread("messi5.jpg")
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 127, 255, 0)
    image, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    img = cv.drawContours(image, contours, 3, (0, 255, 0), 3)

    plt.subplot(1, 2, 1), plt.imshow(im), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(img), plt.title('Contours')
    plt.xticks([]), plt.yticks([])
    plt.show()