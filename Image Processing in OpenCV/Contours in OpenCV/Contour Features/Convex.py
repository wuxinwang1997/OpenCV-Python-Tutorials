# -*- coding: utf-8 -*-
"""
@author: wuxin
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":

    img = cv.imread("Polygon.jpg")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret , binary = cv.threshold(gray, 220, 255, cv.THRESH_BINARY)
    image, contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # 轮廓
    cnt = contours[0]

    # params
    # points 传入的轮廓
    # hull 输出
    # clockwise 方向标志，True为顺时针，False为逆时针
    # returnPoints 默认为True，返回凸包上的点坐标。若为False则返回与凸包点对应的轮廓上的点
    # hull = cv2.convexHull(point[, hull[, clockwise[, returnPoints]]])
    #如果你想获得凸性缺陷，需要把returnPoints 设置为False。以上面的矩形为例，首先我们找到他的轮廓cnt。现在我把returnPoints 设置为True 查找凸包，
    hull = cv.convexHull(cnt)

    print(hull)

    # 现在把returnPoints 设置为False
    # 得到轮廓点的索引
    cnt_index = cv.convexHull(cnt, returnPoints=False)
    print(cnt_index)