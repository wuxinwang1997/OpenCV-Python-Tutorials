# -*- coding: utf-8 -*-
"""
@ author: wuxin
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    filename = "ChessBoard.jpg"
    img = cv.imread(filename)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    gray = np.float32(gray)

    # 输入图像必须是float32，最后一个参数在0.04到0.05之间
    dst = cv.cornerHarris(gray, 2, 3, 0.04)

    # resukt is dilated for marking the corners, not important
    dst = cv.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()] = [0, 0, 255]

    cv.imshow('dst', img)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()