# -*- coding:utf-8 -*-
"""
@author : wuxin
"""

import cv2 as cv
import numpy as np
import glob

if __name__ == '__main__':

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like(0, 0, 0), (1, 0, 0), (2, 0, 0),...,(6, 5, 0)
    objp = np.zeros((6*7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objponits = []  # 3d point in real world space
    imgpoints = []  # 2d point in image plane

    images = glob.glob('*.jpg')

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (7, 6), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objponits.append(objp)

            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            cv.drawChessboardCorners(img, (7, 6), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(500)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objponits, imgpoints, gray.shape[::-1], None, None)

    img = cv.imread('left12.jpg')
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imshow('calib-undistort-result', dst)
    cv.waitKey(500)
    cv.imwrite('calib-undistort-result.png', dst)

    # remapping
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imshow('calib-remapping-result', dst)
    cv.imwrite('calib-remapping-result.png', dst)

    np.savez('../Pose Estimation/B.npz', mtx, dist, rvecs, tvecs)


    mean_error = 0
    for i in range(len(objponits)):
        imgpoints2, _ = cv.projectPoints(objponits[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error

    print('total error: {}'.format(mean_error/len(objponits)))
    cv.waitKey(0)
    cv.destroyAllWindows()
