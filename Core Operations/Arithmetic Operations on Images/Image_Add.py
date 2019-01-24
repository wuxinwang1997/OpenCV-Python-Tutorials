import cv2 as cv
import numpy as np

if __name__ == "__main__":
    # x = np.uint8([250])
    # y = np.uint8([10])
    # print(cv2.add(x, y))
    # print(x + y)
    img1 = cv.imread('ML.jpg')
    img2 = cv.imread('opencv.jpg')

    dst = cv.addWeighted(img1, 0.7, img2, 0.3, 0)

    cv.imshow('dst', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()