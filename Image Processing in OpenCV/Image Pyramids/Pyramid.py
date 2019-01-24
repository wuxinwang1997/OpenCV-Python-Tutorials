# -*- coding: utf-8 -*-
"""
@author: wuxin
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":

    img = cv.imread("messi5.jpg")
    b, g, r = cv.split(img)
    higher_reso = cv.merge([r, g, b])
    lower_reso1 = cv.pyrDown(higher_reso)
    lower_reso2 = cv.pyrDown(lower_reso1)
    higher_reso1 = cv.pyrUp(lower_reso2)
    higher_reso2 = cv.pyrUp(higher_reso1)

    plt.subplot(2,3,1),plt.imshow(higher_reso,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,3,2),plt.imshow(lower_reso1,cmap = 'gray')
    plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,3,3),plt.imshow(lower_reso2,cmap = 'gray')
    plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,3,4),plt.imshow(higher_reso1,cmap = 'gray')
    plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,3,5),plt.imshow(higher_reso2,cmap = 'gray')
    plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
    plt.show()