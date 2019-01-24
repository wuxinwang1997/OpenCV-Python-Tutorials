# -*- coding: utf-8 -*-
"""
@author: wuxin
"""

import cv2 as cv
import numpy as np,sys
from matplotlib import pyplot as plt

if __name__ == "__main__":

    A = cv.imread("Apple.jpg")
    b, g, r = cv.split(A)
    A = cv.merge([r, g, b])
    B = cv.imread("E:/Python/opencv/Picture/Orange.jpg")
    b, g, r = cv.split(B)
    B = cv.merge([r, g, b])

    # generate Gaussian pyramid for apple
    G = A.copy()
    gpA = [G]
    for i in range(6):
        G = cv.pyrDown(G)
        gpA.append(G)

    # generate Gaussian pyramid for orange
    G = B.copy()
    gpB = [G]
    for i in range(6):
        G = cv.pyrDown(G)
        gpB.append(G)

    # generate Laplacian Pyramid for A
    lpA = [gpA[5]]
    for i in range(5, 0, -1):
        size = (gpA[i - 1].shape[1], gpA[i - 1].shape[0])
        print(size)
        GE = cv.pyrUp(gpA[i], dstsize=size)
        L = cv.subtract(gpA[i - 1], GE)
        lpA.append(L)

    # generate Laplacian Pyramid for B
    lpB = [gpB[5]]
    for i in range(5, 0, -1):
        size = (gpB[i - 1].shape[1], gpB[i - 1].shape[0])
        print(size)
        GE = cv.pyrUp(gpB[i], dstsize = size)
        L = cv.subtract(gpB[i-1],GE)
        lpB.append(L)

    # Now add left and right halves of images in each level
    # numpy.hstack(tup)
    # Take a sequence of arrays and stack them horizontally
    # to make a single array
    LS = []
    for la, lb in zip(lpA, lpB):
        rows, cols, dpt = la.shape
        ls = np.hstack((la[:, :int(cols/2)], lb[:, int(cols/2):]))
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1, 6):
        size = (LS[i].shape[1], LS[i].shape[0])
        ls_ = cv.pyrUp(ls_, dstsize=size)
        ls_ = cv.add(ls_, LS[i])

    # image with direct connecting each half
    real = np.hstack((A[:,:int(cols/2)],B[:,int(cols/2):]))

    plt.subplot(221), plt.imshow(A), plt.title("Apple")
    plt.xticks([]), plt.yticks([])
    plt.subplot(222), plt.imshow(B), plt.title("Orange")
    plt.xticks([]), plt.yticks([])
    plt.subplot(223), plt.imshow(real), plt.title("Pyramid_blending2")
    plt.xticks([]), plt.yticks([])
    plt.subplot(224), plt.imshow(ls_), plt.title("Direct_blending")
    plt.xticks([]), plt.yticks([])

    plt.show()