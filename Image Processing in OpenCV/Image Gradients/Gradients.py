
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":

    img = cv.imread("ChessBoard.jpg")
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv2.VC_64F 输出图像的深度（数据类型），可以使用-1，与原图像保持一致 np.uint8
    laplacian = cv.Laplacian(img, cv.CV_64F)
    # 参数 1, 0 为只为在 x 方向求一阶导数，最大可求2阶导数
    sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
    # 参数0,1 为只在y 方向求一阶导数，最大可以求2 阶导数。
    sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)

    plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    plt.show()