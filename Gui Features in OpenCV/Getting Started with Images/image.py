# -*- coding: utf-8 -*-
"""
@author: wuxin
"""

import numpy as np
import cv2 as cv

if __name__ == "__main__":
    #load an color image in grayscale
    img = cv.imread('messi5.jpg', 0)
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv.imshow('image', img)
    k = cv.waitKey(0)&0xFF
    if k == 27:         # wait for ESC key to exit
        cv.destroyAllWindows()
    elif k == ord('s'):
        cv.imwrite('E:/Python/opencv/Picture/messigray.png', img)
        cv.destroyAllWindows()
