# -*- coding:utf-8 -*-
"""
@ author: wuxin
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    x = np.random.randint(25, 100, 25)
    y = np.random.randint(175, 255, 25)
    z = np.hstack((x, y))
    z = z.reshape((50, 1))
    z = np.float32(z)
    plt.hist(z, 256, [0, 256])

    # Define criteria = {type, max_iter = 10, epsilon = 1.0}
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Set flags (Just to aviud line break in the code)
    flags = cv.KMEANS_RANDOM_CENTERS

    # APPLY KMeans
    compactness, labels, centers = cv.kmeans(z, 2, None,criteria, 10, flags=flags)

    A = z[labels==0]
    B = z[labels==1]

    # Now plot 'A' in red, 'B' in blue, 'center' in yellow
    plt.hist(A, 256, [0, 256], color='r')
    plt.hist(B, 256, [0, 256], color='b')
    plt.hist(centers, 32, [0, 256], color='y')
    plt.show()