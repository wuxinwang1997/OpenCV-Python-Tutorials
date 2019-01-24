# -*- coding: utf-8 -*-
"""
@author: wuxin
"""
import cv2 as cv
import numpy as np

if __name__ == "__main__":
    cap = cv.VideoCapture(0)

    while(1):

        # 获取每一帧
        ret, frame = cap.read()

        # 转换到HSV
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # 设定蓝色的阈值
        lower_blue = np.array([110, 50, 50])
        upper_blue = np.array([130, 255, 255])

        # 根据阈值构建掩模
        mask = cv.inRange(hsv, lower_blue, upper_blue)

        # 对原图像和掩模进行位运算
        res = cv.bitwise_and(frame, frame, mask=mask)

        # 显示图像
        cv.imshow('frame', frame)
        cv.imshow('mask', mask)
        cv.imshow('res', res)
        k = cv.waitKey(5) & 0xFF
        if k == ord('q'):
            break
    # 关闭窗口
    cv.destroyAllWindows()