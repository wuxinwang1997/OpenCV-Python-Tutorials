# -*- coding: utf-8 -*-
"""
@author: wuxin
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":

    img = cv2.imread('bilateral.jpg')
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    #cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)
    #d – Diameter of each pixel neighborhood that is used during filtering.
    # If it is non-positive, it is computed from sigmaSpace
    #9 邻域直径，两个75 分别是空间高斯函数标准差，灰度值相似性高斯函数标准差
    blur = cv2.bilateralFilter(img, 9, 75, 75)

    plt.subplot(121), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(blur), plt.title('BilateralFilter')
    plt.xticks([]), plt.yticks([])
    plt.show()