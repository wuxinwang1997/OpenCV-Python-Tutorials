# -*- coding: utf-8 -*-
"""
@author: wuxin
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":

    img = cv.imread("Polygon.jpg")
    img1 = np.copy(img)

    rows, cols = img.shape[:2]

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret , binary = cv.threshold(gray, 220, 255, cv.THRESH_BINARY)
    image, contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # 轮廓
    cnt = contours[0]

    #cv2.fitLine(points, distType, param, reps, aeps[, line ]) → line
    #points – Input vector of 2D or 3D points, stored in std::vector<> or Mat.
    #line – Output line parameters. In case of 2D fitting, it should be a vector of
    #4 elements (likeVec4f) - (vx, vy, x0, y0), where (vx, vy) is a normalized
    #vector collinear to the line and (x0, y0) is a point on the line. In case of
    #3D fitting, it should be a vector of 6 elements (like Vec6f) - (vx, vy, vz,
    #x0, y0, z0), where (vx, vy, vz) is a normalized vector collinear to the line
    #and (x0, y0, z0) is a point on the line.
    #distType – Distance used by the M-estimator
    #distType=CV_DIST_L2
    #ρ(r) = r2 /2 (the simplest and the fastest least-squares method)
    #param – Numerical parameter ( C ) for some types of distances. If it is 0, an optimal value
    #is chosen.
    #reps – Sufficient accuracy for the radius (distance between the coordinate origin and the
    #line).
    #aeps – Sufficient accuracy for the angle. 0.01 would be a good default value for reps and
    #aeps.
    [vx,vy,x,y] = cv.fitLine(cnt, cv.DIST_L2,0,0.01,0.01)
    lefty = int((-x*vy/vx) + y)
    righty = int(((cols-x)*vy/vx)+y)
    img1 = cv.line(img1,(cols-1,righty),(0,lefty),(0,255,0),2)

    plt.subplot(121), plt.imshow(img)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img1)
    plt.xticks([]), plt.yticks([])
    plt.show()