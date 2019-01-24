# -*- coding:utf-8 -*-
"""
@ author: wuxin
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    # Load the datda, converters convert the letter to a number
    data = np.loadtxt('letter-recognition.data', dtype='float32', delimiter=',',
                      converters={0: lambda ch: ord(ch)-ord('A')})

    # split the data to two, 10000 each for train and test
    train, test = np.vsplit(data, 2)

    # split trainData and testData to features and responses
    responses, trainData = np.hsplit(train, [1])
    labels, testData = np.hsplit(test, [1])

    # Initiate the kNn, classify, measure accuracy.
    knn = cv.ml.KNearest_create()
    knn.train(trainData, cv.ml.ROW_SAMPLE, responses)
    ret, result, neighbours, dost = knn.findNearest(testData, k=5)

    correct = np.count_nonzero(result == labels)
    accuracy = correct*100.0/10000
    print(accuracy)