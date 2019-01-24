import cv2 as cv
import numpy as np

if __name__ == "__main__":

    img1 = cv.imread("Bounding.jpg", 0)
    img2 = cv.imread("bound.jpg", 0)

    ret1, thresh1 = cv.threshold(img1, 127, 255, 0)
    ret2, thresh2 = cv.threshold(img2, 127, 255, 0)

    iamge, contours, hierarchy = cv.findContours(thresh1, 2, 1)
    cnt1 = contours[0]
    iamge, contours, hierarchy = cv.findContours(thresh2, 2, 1)
    cnt2 = contours[0]

    ret = cv.matchShapes(cnt1, cnt2, 1, 0.0)
    print(ret)

    cv.imshow('img1', img1)
    cv.imshow('img2', img2)
    cv.waitKey(0)
    cv.destroyAllWindows()
