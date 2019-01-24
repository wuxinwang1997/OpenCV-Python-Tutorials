# -*- coding: utf-8 -*-
"""
@author: wuxin
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    img = cv.imread("messi5.jpg", 0)
    edges = cv.Canny(img, 100, 200)

    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Image')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray'), plt.title('Edge Image')
    plt.xticks([]), plt.yticks([])

    plt.show()
