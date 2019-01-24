# -*- coding: utf-8 -*-
"""
@author: wuxin
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":

    img = cv.imread("j.jpg")
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv.erode(img, kernel, iterations=1)
    dilation = cv.dilate(img, kernel,iterations = 1)
    img_open = cv.imread("j_open.jpg")
    opening = cv.morphologyEx(img_open, cv.MORPH_OPEN, kernel)
    img_close = cv.imread("j_close.jpg")
    closing = cv.morphologyEx(img_close, cv.MORPH_CLOSE, kernel)
    gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
    kernel1 = np.ones((9, 9), np.uint8)
    tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel1)
    blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel1)


    plt.subplot(4, 3, 1), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(4, 3, 2), plt.imshow(erosion), plt.title('Erode')
    plt.xticks([]), plt.yticks([])
    plt.subplot(4, 3, 3), plt.imshow(dilation), plt.title('Dilation')
    plt.xticks([]), plt.yticks([])
    plt.subplot(4, 3, 4), plt.imshow(img_open), plt.title('img_open')
    plt.xticks([]), plt.yticks([])
    plt.subplot(4, 3, 5), plt.imshow(opening), plt.title('Opening')
    plt.xticks([]), plt.yticks([])
    plt.subplot(4, 3, 6), plt.imshow(img_close), plt.title('img_close')
    plt.xticks([]), plt.yticks([])
    plt.subplot(4, 3, 7), plt.imshow(closing), plt.title('Closing')
    plt.xticks([]), plt.yticks([])
    plt.subplot(4, 3, 8), plt.imshow(gradient), plt.title('Gradient')
    plt.xticks([]), plt.yticks([])
    plt.subplot(4, 3, 9), plt.imshow(tophat), plt.title('Tophat')
    plt.xticks([]), plt.yticks([])
    plt.subplot(4, 3, 10), plt.imshow(tophat), plt.title('Tophat')
    plt.xticks([]), plt.yticks([])
    plt.show()