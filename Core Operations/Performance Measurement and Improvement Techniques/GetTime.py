import cv2 as cv
import numpy as np
import time

if __name__ == "__main__":

    img1 = cv.imread('messi5.jpg')

    e1 = cv.getTickCount()
    # e1 = time.time()
    # your code execution
    for i in range(5, 49, 2):
        img1 = cv.medianBlur(img1, i)
    e2 = cv.getTickCount()
    # e2 = time.time()
    time = (e2 - e1)/cv.getTickFrequency()
    print(time)