# -*- coding:utf-8 -*-
"""
@ author: wuxin
"""

import cv2 as cv
import numpy as np

MIN_MATCH_COUNT = 10
# FLANN parameters
FLANN_INDEX_KDTREE = 1

if __name__ == '__main__':

    img1 = cv.imread('img1.png', 0)             # queryingImage
    img1_match = cv.imread('img1_match.png', 0) #trainImage

    # Initiate SIFT detector
    sift =cv.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp1_match, des1_match = sift.detectAndCompute(img1_match, None)

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des1_match, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        # get the coordinates of keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1_match[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # 第三个参数 Method used to computed a homography matrix. The following methods are passible:
        # 0 - a regular method using all the points
        # CV_RANSAC - RANSAC-based robust method
        # CV_LMEDS - Least-Median robust method
        # 第四个参数取值范围在 1 到 10，拒绝一个点对的阈值。原图像的点经过变换后点与目标图像上对应点的误差
        # 超过误差就认为为是 outlier
        # 返回值中 M 为变换矩阵
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        # 获得原图像的高和宽
        h, w = img1.shape

        # 使用得到的变换矩阵对原图像的四个角进行变换，获得目标图像上对应的组表
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)

        # 原图像为灰度图
        img2 = cv.polylines(img1_match, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

    else:
        print('Not enough matches are found - {}/{}'.format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=2)

    result = cv.drawMatches(img1, kp1, img1_match, kp1_match, good, None, **draw_params)

    cv.imshow('Feature Matching + Homography to ind Objects', result)
    cv.waitKey(0)
    cv.destroyAllWindows()
