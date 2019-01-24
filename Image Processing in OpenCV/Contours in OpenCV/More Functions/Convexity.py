# -*- dofing: utf-8 -*-
"""
@ author: wuxin
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":

    img = cv.imread("Bounding.jpg")
    img1 = np.copy(img)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(gray, 127, 255, 0)
    image, contours, hierarchy = cv.findContours(thresh,2,1)
    # 轮廓
    cnt = contours[0]

    hull = cv.convexHull(cnt, returnPoints = False)
    defects = cv.convexityDefects(cnt, hull)

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[s][0])
        far = tuple(cnt[f][0])
        cv.line(img1, start, end, [0, 255, 0], 2)
        cv.circle(img1, far, 5, [0, 0, 255], -1)

    cv.imshow('img',img)
    cv.imshow('img1',img1)
    cv.waitKey(0)
    cv.destroyAllWindows()