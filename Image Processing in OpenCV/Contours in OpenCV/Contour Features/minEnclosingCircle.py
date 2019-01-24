# -*- coding: utf-8 -*-
"""
@author: wuxin
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":

    img = cv.imread("Polygon.jpg")
    img1 = np.copy(img)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret , binary = cv.threshold(gray, 220, 255, cv.THRESH_BINARY)
    image, contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # 轮廓
    cnt = contours[0]

    (x, y), radius = cv.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    img1 = cv.circle(img1, center, radius, (0, 255, 0), 2)

    plt.subplot(121), plt.imshow(img)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img1)
    plt.xticks([]), plt.yticks([])
    plt.show()