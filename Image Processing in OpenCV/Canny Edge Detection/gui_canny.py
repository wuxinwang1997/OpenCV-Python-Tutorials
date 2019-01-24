"""
@author: wuxin
"""
import cv2
import numpy as np

def nothing(x):
    pass

if __name__ == "__main__":
    # 创建一副黑色图像
    img = cv2.imread("messi5.jpg", 0)

    cv2.namedWindow('image')

    cv2.createTrackbar('minVal', 'image', 0, 255, nothing)
    cv2.createTrackbar('maxVal', 'image', 0, 255, nothing)


    while(1):
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        minVal = cv2.getTrackbarPos('minVal', 'image')
        maxVal = cv2.getTrackbarPos('maxVal', 'image')

        edges = cv2.Canny(img, minVal, maxVal)
        cv2.imshow('edges', edges)

    cv2.destroyAllWindows()