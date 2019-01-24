# -*- coding: utf-8 -*-
"""
@author: wuxin
"""
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":

    img = cv.imread('sudokusmall.png')
    rows,cols,ch=img.shape
    pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
    pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
    M=cv.getPerspectiveTransform(pts1,pts2)
    dst=cv.warpPerspective(img,M,(300,300))

    plt.subplot(121), plt.imshow(img), plt.title('Input')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img), plt.title('Output')
    plt.xticks([]), plt.yticks([])
    plt.show()
    # 与原始图片尺寸不一样