import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

if __name__ == "__main__":

    img = cv.imread('home.png', 0)
    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()