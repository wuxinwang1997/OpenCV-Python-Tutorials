# -*- coding: utf-8 -*-
"""
@author: wuxin
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    img = cv.imread("Polygon.jpg")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret , binary = cv.threshold(gray, 220, 255, cv.THRESH_BINARY)
    image, contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # 轮廓
    cnt = contours

    img1 = np.copy(img)
    epsilon1 = 0.1 * cv.arcLength(cnt[0], True)
    approx1 = cv.approxPolyDP(cnt[0], epsilon1, True)
    img1 = cv.drawContours(img1, cnt ,-1, (255,0,0), 3)

    img2 = np.copy(img)
    epsilon2 = 0.01 * cv.arcLength(cnt[0], True)
    approx2 = cv.approxPolyDP(cnt[0], epsilon2, True)
    img2 = cv.drawContours(img2, approx2 ,-1, (255,0,0), 3)

    plt.subplot(1, 3, 1), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 2), plt.imshow(img1), plt.title('epsilon = 10%')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 3), plt.imshow(img2), plt.title('epsilon = 1%')
    plt.xticks([]), plt.yticks([])

    plt.show()