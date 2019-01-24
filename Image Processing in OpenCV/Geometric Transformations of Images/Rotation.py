# -*- coding: utf-8 -*-
"""
@author: wuxin
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":

    img = cv.imread('messi5.jpg', 0)

    rows,cols=img.shape
    # 这里的第一个参数为旋转中心，第二个为旋转角度，第三个为旋转后的缩放因子
    # 可以通过设置旋转中心，缩放因子，以及窗口大小来防止旋转后超出边界的问题
    M=cv.getRotationMatrix2D((cols/2, rows/2), 90, 2.0)
    # 第三个参数是输出图像的尺寸中心
    dst=cv.warpAffine(img, M, (2*cols, 2*rows))

    plt.subplot(121), plt.imshow(img), plt.title('Input')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(dst), plt.title('Output')
    plt.xticks([]), plt.yticks([])
    plt.show()