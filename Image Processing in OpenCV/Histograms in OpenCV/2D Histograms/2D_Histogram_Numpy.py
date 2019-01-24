import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

if __name__ == "__main__":

    img = cv.imread('home.png')
    b, g, r = cv.split(img)
    img = cv.merge([r, g, b])
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    hst, xbins, ybins = np.histogram2d(h.ravel(), s.ravel(), [180, 256], [[0, 180], [0, 256]])
    plt.subplot(121), plt.imshow(img), plt.title('img')
    plt.subplot(122), plt.plot(hst), plt.title('2D-hst')
    plt.show()