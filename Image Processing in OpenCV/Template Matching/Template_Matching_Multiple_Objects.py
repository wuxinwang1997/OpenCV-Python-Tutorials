import cv2 as cv
import numpy as np

if __name__ == "__main__":

    img_rgb = cv.imread('mario.png')
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    template = cv.imread('mario_coin.jpg', 0)
    w, h = template.shape[::-1]

    res = cv.matchTemplate(img_gray, template, cv.TM_CCORR_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv.rectangle(img_rgb, pt, (pt[0] + 2, pt[1] + h), (0, 0, 255), 2)
    cv.imshow('res', img_rgb)
    cv.waitKey(0)
    cv.drawKeypoints()