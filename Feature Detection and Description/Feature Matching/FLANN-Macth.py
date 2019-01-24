# -*- coding:utf-8 -*-
"""
@ author: wuxin
"""

import cv2 as cv
import numpy as np

if __name__ == '__main__':

    img1 = cv.imread('img1.png', 0)             # queryingImage
    img1_match = cv.imread('img1_match.png', 0)       # trainImage

    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp1_match, des1_match = sift.detectAndCompute(img1_match, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)           # or pass empty dictionary

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des1_match, 2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i, (m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)

    result = cv.drawMatchesKnn(img1, kp1, img1_match, kp1_match, matches, None, **draw_params)

    cv.imshow('FLANN-Match', result)
    cv.waitKey(0)
    cv.destroyAllWindows()