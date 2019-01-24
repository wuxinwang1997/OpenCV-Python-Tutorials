# -*- coding:utf-8 -*-
"""
@ author: wuxin
"""

import cv2 as cv
import numpy as np

if __name__ == '__main__':
    img1 = cv.imread('img1.png', 0)             # queryImage
    img1_match = cv.imread('img1_match.png', 0) # trainImage

    # Initiate ORB detector
    orb = cv.ORB_create()

    # find the keypoints and descriptors witj ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp1_match, des1_match = orb.detectAndCompute(img1_match, None)

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des1_match)

    # Sort them in the order of their distance
    matches = sorted(matches, key = lambda x:x.distance)

    # Draw first 10 matches.
    result = cv.drawMatches(img1, kp1, img1_match, kp1_match, matches[:20], None, flags=2)

    cv.imshow('ORB Match', result)
    cv.waitKey(0)
    cv.destroyAllWindows()