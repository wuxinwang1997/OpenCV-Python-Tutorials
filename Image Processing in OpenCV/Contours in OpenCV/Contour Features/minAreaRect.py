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

    rect = cv.minAreaRect(cnt)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    img1 = cv.drawContours(img1,[box],0,(0,0,255),2)
    # img1 = cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)

    plt.subplot(121), plt.imshow(img)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img1)
    plt.xticks([]), plt.yticks([])
    plt.show()