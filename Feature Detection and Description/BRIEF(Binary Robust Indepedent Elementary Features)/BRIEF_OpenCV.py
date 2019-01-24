# -*- coding:utf-8 -*-
"""
@ author: wuxin
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    img = cv.imread('simple.jpg', 0)

    # Initiate FAST detector
    star = cv.xfeatures2d.StarDetector_create()

    # Initiate BRIEF extractor
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

    # find the keypoints with STAR
    kp = star.detect(img, None)

    # compute the descriptors woth BRIEF
    kp, des = brief.compute(img, kp)

    print(brief.descriptorSize())
    print(des.shape)