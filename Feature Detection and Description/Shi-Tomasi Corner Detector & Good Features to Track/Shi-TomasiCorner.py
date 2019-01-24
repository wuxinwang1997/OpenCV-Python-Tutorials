# -*- coding: utf-8 -*_
"""
@ author :wuxin
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    img = cv.imread('blox.jpg')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    corners = cv.goodFeaturesToTrack(gray, 25, 0.01, 10)
    # 返回的结果是[[311.,250.]]两层括号的数组
    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()
        cv.circle(img, (x,y), 5, 0, -1)

    plt.imshow(img), plt.show()