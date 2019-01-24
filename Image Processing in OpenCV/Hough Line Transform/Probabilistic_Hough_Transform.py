# -*- coding:utf-8 -*-
"""
@ author:wuxin
"""

import cv2 as cv
import numpy as np

if __name__ == "__main__":

    img = cv.imread('ChessBoard.jpg')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    minLineLength = 100
    maxLineGap = 10
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap)
    for j in range(len(lines)):
        for x1,y1,x2,y2 in lines[j]:
            cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv.imshow('houghlines2.jpg', img)
    cv.waitKey(0)
    cv.destroyAllWindows()