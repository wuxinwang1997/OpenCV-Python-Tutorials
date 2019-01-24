# -*- coding:utf-8 -*-
"""
@ author: wuxin
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    # Feature set containing (x, y) values of 25 known/training data
    trainDData = np.random.randint(0, 100, (25, 2)).astype(np.float32)

    # Labels each one either Red or Blue with number 0 and 1
    responses = np.random.randint(0, 2, (25, 1)).astype(np.float32)

    # Take Red familes and plot them
    red = trainDData[responses.ravel() == 0]
    plt.scatter(red[:, 0], red[:, 1], 80, 'r', '^')

    # Take Blue families and plot them
    blue = trainDData[responses.ravel() == 1]
    plt.scatter(blue[:, 0], blue[:, 1], 80, 'b', 's')

    newcomer = np.random.randint(0, 100, (1, 2)).astype(np.float32)
    plt.scatter(newcomer[:, 0], newcomer[:, 1], 80, 'g', 'o')

    knn = cv.ml.KNearest_create()
    knn.train(trainDData, cv.ml.ROW_SAMPLE, responses)
    ret, results, neighbours, dist = knn.findNearest(newcomer, 3)

    print('result: {}\n'.format(results))
    print('neighbours: {}\n'.format(neighbours))
    print('distance: {}\n'.format(dist))

    plt.show()
