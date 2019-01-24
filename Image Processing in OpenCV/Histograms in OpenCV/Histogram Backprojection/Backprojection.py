import numpy as np
import cv2 as cv
if __name__ == "__main__":

    roi = cv.imread('red_rose.jpg')
    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    target = cv.imread('rose.jpg')
    hsvt = cv.cvtColor(target,cv.COLOR_BGR2HSV)
    # calculating object histogram
    roihist = cv.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
    # normalize histogram and apply backprojection
    cv.normalize(roihist,roihist,0,255,cv.NORM_MINMAX)
    dst = cv.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)
    # Now convolute with circular disc
    disc = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    cv.filter2D(dst,-1,disc,dst)
    # threshold and binary AND
    ret,thresh = cv.threshold(dst,50,255,0)
    thresh = cv.merge((thresh,thresh,thresh))
    res = cv.bitwise_and(target,thresh)
    res = np.vstack((target,thresh,res))
    cv.imwrite('res.jpg',res)