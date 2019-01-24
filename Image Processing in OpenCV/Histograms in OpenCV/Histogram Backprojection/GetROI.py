import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

if __name__ == "__main__":

    roi = cv.imread("roi.png")
    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

    target = cv.imread("messi5.jpg")
    hsvt = cv.cvtColor(target, cv.COLOR_BGR2HSV)

    # calculating object histogram
    roihist = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

    # normalize histgoram and apply backprojection
    # 归一化：原始图像，结果图像，映射到结果图像中的最小值，最大值，归一化类型
    # cv2.NORM_MINMAX 对数组的所有值今昔那个转化，使它们线性映射到最小值和最大值之间
    # 归一化后的直方图便于显示，归一化后就成了 0 到 255 之间的数了
    cv.normalize(roihist, roihist, 0, 255, cv.NORM_MINMAX)
    dst = cv.calcBackProject([hsvt], [0, 1], roihist, [0, 180, 0, 255], 1)

    # Now canvlute with circular disc
    # 此处卷积可以把分数的点连在一起
    disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    dst = cv.filter2D(dst, -1, disc)

    # threshold and binary AND
    ret, thresh = cv.threshold(dst, 50, 255, 0)
    # 别忘了是三通道图像
    thresh = cv.merge((thresh, thresh, thresh))
    # 按位操作
    res = cv.bitwise_and(target, thresh)

    res = np.hstack((target, thresh, res))

    plt.imshow(res), plt.title('res')
    plt.show()