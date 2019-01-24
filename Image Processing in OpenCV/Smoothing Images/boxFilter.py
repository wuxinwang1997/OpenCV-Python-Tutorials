# -*- coding: utf-8 -*-
"""
@author: wuxin
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":

    img = cv2.imread('opencv.jpg')
    img = img.astype(np.float32) / 255.0

    boxFilter = cv2.boxFilter(img, 5, (5, 5), normalize=False)

    plt.subplot(121), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(boxFilter), plt.title('BoxFiltered')
    plt.xticks([]), plt.yticks([])
    plt.show()