# -*- coding: utf-8 -*-
"""
@ author: wuxin
"""

import cv2 as cv
import numpy as np

if __name__ == "__main__":
    filename = "chessboard2.jpg"
    img = cv.imread(filename)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # find Harris corners
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, 2, 3, 0.04)
    dst = cv.dilate(dst, None)
    ret, dst = cv.threshold(dst, 0.01*dst.max(), 255, 0)
    dst = np.uint8(dst)

    # find cenraids
    # connectedComponentsWithStats(InputArray image, OutputArray labels, OutputArray stats, OutputArray centroids, int connectivity = 8, int ltype=CV_32S)
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    # Python: cv.ornerSubPix(image, corners, winSize, zeroZone, criteria)
    # zroZone - Half of the size of the dad region in the middle of the search zone
    # over which the sunation in the autocorrelation matrix. The value of (-1,-1)
    # indicates that there is no such a size
    # 返回值由角点坐标组成的一个数组（而非图像）
    corners = cv.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

    # Now draw them
    res = np.hstack((centroids, corners))
    # np.int0可以用来神略小数点后的数字（非四舍五入)
    res = np.int0(res)
    img[res[:,1], res[:,0]] = [0, 0, 255]
    img[res[:,3], res[:,2]] = [0, 255, 0]

    cv.imshow('result', img)
    cv.waitKey(0)
    cv.destroyAllWindows()