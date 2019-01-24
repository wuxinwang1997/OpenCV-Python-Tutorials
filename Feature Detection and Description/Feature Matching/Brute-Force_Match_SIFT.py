# -*- coding:utf-8 -*-
"""
@author: wuxin
"""

import cv2 as cv
import numpy as np

if __name__ == '__main__':

    img2 = cv.imread('img2.png', 0)
    img2_match = cv.imread('img2_match.png', 0)

    # Inintiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp2, des2 = sift.detectAndCompute(img2, None)
    kp2_match, des2_match = sift.detectAndCompute(img2_match, None)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des2, des2_match, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    # cv.drawMatchesKnn expects list of lists as mathches
    result = cv.drawMatchesKnn(img2, kp2, img2_match, kp2_match, good, None, flags=2)

    cv.imshow('Brute-Force Matching with SIFT DEescriptors and Ratio Test', result)
    cv.waitKey(0)
    cv.drawMatchesKnn()