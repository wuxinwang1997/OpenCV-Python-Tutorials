# -*- coding:utf-8 -*-
"""
@author: wuxin
"""

import cv2 as cv
import numpy as np

if __name__ == "__main__":

    img = cv.imread('simple.jpg', 0)

    # Initiate STAR detector
    orb = cv.ORB_create()

    # find the keypoints with ORB
    kp = orb.detect(img, None)

    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)

    # draw only keypoints location, not size and orientation
    img2 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))
    cv.imshow('ORB', img2)
    cv.waitKey(0)
    cv.destroyAllWindows()