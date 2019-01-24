import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    img = cv.imread('messi5.jpg')
    b,g,r = cv.split(img)
    img2 = cv.merge([r, g, b])
    plt.subplot(121);plt.imshow(img)  # expects distorted color
    plt.subplot(122);plt.imshow(img2) # expect true color
    plt.show()

    cv.imshow('bgr image', img)    # expects true color
    cv.imshow('rgb image', img2)   # exprcts disorted color
    cv.waitKey(0)
    cv.destroyAllWindows()