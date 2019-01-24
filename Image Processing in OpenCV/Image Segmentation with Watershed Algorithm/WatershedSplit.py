# -*- coding:utf-8 -*-
"""
@ author:wuxin
"""
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":

    img = cv.imread('water_coins.png')
    b, g, r = cv.split(img)
    img = cv.merge([r, g, b])
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    plt.subplot(121), plt.title('Original'), plt.imshow(img)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.title('Ostu'), plt.imshow(thresh)
    plt.xticks([]), plt.yticks([])
    plt.show()

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

    # Finding sure foreground area
    sure_bg = cv.dilate(opening, kernel, iterations=3)


    # 距离变换的基本含义是计算一个图像中非零像素点到最近的零像素点的距离，也就是到零像素点的最短距离
    # 最常见的距离变换算法就是通过连接的服饰操作来实现，腐蚀操作的停止条件是所有前景像素都被完全
    # 腐蚀。这样根据腐蚀的先后顺序，我们就得到各个前景像素点到前景中心骨架像素点的
    # 距离。根据各个像素点的距离值，设置为不同的灰度值，这样就完成了二值图像的距离变换
    # cv.distanceTransform(src, distanceType, maskSize)
    dist_transform = cv.distanceTransform(opening, distanceType=cv.DIST_L2, maskSize=5)
    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unkown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    plt.subplot(121), plt.imshow(sure_bg), plt.title('腐蚀背景提取')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(sure_fg), plt.title('腐蚀前景提取')
    plt.xticks([]), plt.yticks([])
    plt.show()

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    plt.imshow(img), plt.title('result')
    plt.xticks([]), plt.yticks([])
    plt.show()