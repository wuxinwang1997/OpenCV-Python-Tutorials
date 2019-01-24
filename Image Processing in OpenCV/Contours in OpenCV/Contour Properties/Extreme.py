# -*- dofing: utf-8 -*-
"""
@ author: wuxin
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":

    img = cv.imread("mostpoints.png")
    img1 = np.copy(img)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret , binary = cv.threshold(gray, 220, 255, cv.THRESH_BINARY)
    image, contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # 轮廓
    cnt = contours[0]

    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])

    img1 = cv.circle(img1, leftmost, 20, [0,255,0], -1)
    img1 = cv.circle(img1, rightmost, 20, [0,255,0], -1)
    img1 = cv.circle(img1, topmost, 20, [0,255,0], -1)
    img1 = cv.circle(img1, bottommost, 20, [0,255,0], -1)

    plt.subplot(121), plt.imshow(img)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img1)
    plt.xticks([]), plt.yticks([])
    plt.show()