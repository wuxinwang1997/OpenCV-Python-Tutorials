import cv2 as cv
import numpy as np

if __name__ == "__main__":

    img = cv.imread('messi5.jpg')
    px = img[100, 100]
    print(px)
    blue = img[100, 100, 0]
    print(blue)
    img[100, 100] = [255, 255, 255]
    print(img[100, 100])
    print(img.item(10, 10, 2))
    img.itemset((10, 10, 2), 100)
    print(img.item(10, 10, 2))
    print(img.shape)
    print(img.size)
    print(img.dtype)
    ball = img[280:340, 330:390]
    img[273:333, 100:160] = ball
    b, g, r = cv.split(img)
    img = cv.merge((b, g, r))
    b = img[:, :, 0]
    img[:, :, 2] = 0
