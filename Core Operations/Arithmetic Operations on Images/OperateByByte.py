import cv2 as cv
import numpy as np

if __name__ == "__main__":

    # 加载图像
    img1 = cv.imread('messi5.jpg')
    img2 = cv.imread('opencv.jpg')

    # I want to put logo on top left corner, So I create a ROI
    rows, cols, channels = img2.shape
    roi = img1[0:rows, 0:cols]

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(img2gray, 175, 255, cv.THRESH_BINARY)
    mask_inv = cv.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    # 取 roi 中与 mask 中不为零的值对应的像素的值，其他值为 0
    # 注意这里必须有 mask=mask 胡总和 mask=mask_inv，其中的mask=不饿能忽略
    img1_bg = cv.bitwise_and(roi, roi, mask=mask)
    # 取 roi 中与 mask_inv 中不为零的值对应的像素的值，其他值为 0
    # Take only region of logo from logo image
    img2_fg = cv.bitwise_and(img2, img2, mask=mask_inv)

    # Put logo in ROI and modify the main image
    dst = cv.add(img1_bg, img2_fg)
    img1[0:rows, 0:cols] = dst

    cv.imshow('res', img1)
    cv.waitKey(0)
    cv.destroyAllWindows()