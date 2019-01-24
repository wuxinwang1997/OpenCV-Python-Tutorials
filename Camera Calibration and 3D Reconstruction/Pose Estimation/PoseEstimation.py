# -*- coding:utf-8 -*-
"""
@author: wuxin
"""

import cv2 as cv
import numpy as np
import glob


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img


def drawCube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground flooe in green
    img = cv.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 3)

    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img


if __name__ == '__main__':

    # Load previously saved data
    with np.load('B.npz') as X:
        mtx, dist, _, _ = [X[i] for i in ('arr_0.npy', 'arr_1.npy', 'arr_2.npy', 'arr_3.npy')]

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
    axisCube = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],
                           [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])

    for fname in glob.glob('left*.jpg'):
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (7, 6), None)

        if ret == True:
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Find the rotation and translation vectors
            ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)

            # project 3D points to image plane
            imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)

            img = draw(img, corners2, imgpts)
            cv.imshow('Pose Estimation', img)
            imgptsCube, jacCube = cv.projectPoints(axisCube, rvecs, tvecs, mtx, dist)
            imgCube = drawCube(img, corners2, imgptsCube)
            cv.imshow('Pose Estimation Cube', imgCube)
            k = cv.waitKey(0) & 0xff
            if k == 'c':
                continue

    cv.destroyWindow()
