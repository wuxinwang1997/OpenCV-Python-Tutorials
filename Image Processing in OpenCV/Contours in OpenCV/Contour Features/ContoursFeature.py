# -*- coding: utf-8 -*-
"""
@author: wuxin
"""

import cv2 as cv
import numpy as np

if __name__ == "__main__":

    img = cv.imread("image.jpg", 0)
    ret, thresh = cv.threshold(img, 127, 255, 0)
    image, contours, hierarchy = cv.findContours(thresh, 1, 2)

    # 轮廓
    cnt = contours[0]
    M = cv.moments(cnt)
    print('M:', M)
    # 用于计算重心
    cx = int(M['m10']/M['m00'])
    cy = int(M['m10']/M['m00'])
    print('cx:', cx, 'cy:', cy)
    # 获取轮廓包含面积
    area = cv.contourArea(cnt)
    print('area:',area)
    # 获取轮廓周长
    perimeter = cv.arcLength(cnt, True)
    print('perimeter:', perimeter)