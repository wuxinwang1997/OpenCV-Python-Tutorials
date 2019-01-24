# -*- coding:utf-8 -*_
"""
@ author: wuxin
"""

import cv2 as cv
import numpy as np

if __name__ == "__main__":

    img = cv.imread('simple.jpg', 0)

    # Initiate FAST object with default values
    fast = cv.FastFeatureDetector_create()

    # find and draw teh keypoints
    kp = fast.detect(img, None)
    img2 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))

    # Print all default params
    print("Thresold: {}".format(fast.getThreshold()))
    print("nonmaxSuppression: {}".format(fast.getNonmaxSuppression()))
    print("neighborhood: {}".format(fast.getType()))
    print("Total Keypoints with nonmaxSuppression: {}".format(len(kp)))

    cv.imshow('fast_true', img2)

    # Disable nonmaxSuppression
    fast.setNonmaxSuppression(0)
    kp = fast.detect(img, None)

    print("Total Keypoints without nonmaxSuppression: {}".format(len(kp)))

    img3 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))

    cv.imshow('fast_false', img3)
    cv.waitKey(0)
    cv.destroyAllWindows()
