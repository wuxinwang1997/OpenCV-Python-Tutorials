import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

if __name__ == "__main__":

    # roi is the object or region of object we need to find
    roi = cv.imread('red_rose.jpg')
    b, g, r = cv.split(roi)
    roi = cv.merge([r, g, b])
    hsv = cv.cvtColor(roi,cv.COLOR_BGR2HSV)
    plt.subplot(221), plt.imshow(roi)
    plt.subplot(222), plt.imshow(hsv)
    #target is the image we search in
    target = cv.imread('rose.jpg')
    b, g, r = cv.split(target)
    target = cv.merge([r, g, b])
    hsvt = cv.cvtColor(target,cv.COLOR_BGR2HSV)
    plt.subplot(223), plt.imshow(target)
    plt.subplot(224), plt.imshow(hsvt)
    plt.show()

    # Find the histograms using calcHist. Can be done with np.histogram2d also
    M = cv.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
    I = cv.calcHist([hsvt],[0, 1], None, [180, 256], [0, 180, 0, 256] )

    #计算比值：R = M/I 。反向投影R，也就是根据R 这个”调色板“创建一副新的图像，其中的每一个像素代表
    # 这个点就是目标的概率。例如B (x; y) =R[h (x; y) ; s (x; y)]，其中h 为点（x，y）处的hue 值，
    # s 为点（x，y）处的saturation 值。最后加入再一个条件B (x; y) = min [B (x; y) ; 1]。
    R = M / (I+ np.finfo(float).eps) # 避免出现分母为 0 的情况
    h, s, v = cv.split(hsvt)
    B = R[h.ravel(), s.ravel()]
    B = np.minimum(B, 1)
    B = B.reshape(hsvt.shape[:2])

    #现在使用一个圆盘算子做卷积，B = D  B，其中D 为卷积核
    disc = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    cv.filter2D(B,-1,disc,B)
    B = np.uint8(B)
    cv.normalize(B,B,0,255,cv.NORM_MINMAX)

    ret,thresh = cv.threshold(B,50,255,0)
    res = cv.bitwise_and(target,target,mask=thresh)
    plt.imshow(res)
    plt.show()

    # 跑官方demo没成功，代码不全 参考:
    # CSDN：https://www.baidu.com/link?
    # url=fOD5PLpTYPUYrsSrRbVRUMkozGJdlQ
    # TcSaViUkoPzbyDhiHWKuE5TAqqMcEnW_r9
    # j2WMrC_b_bQlWLjculTMj7V1I7304zg5RJ
    # T9t6mgHoC&wd=&eqid=ff84535c00005af
    # 2000000035c011e74