import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

if __name__ == "__main__":

    img = cv.imread('wiki.png', 0)
    hist,bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    plt.plot(cdf_normalized, color = 'b')
    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()

    # 构建Numpy 掩模数组，cdf 为原数组，当数组元素为0 时，掩盖（计算时被忽略）。
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    # 对被掩盖的元素赋值，这里赋值为0
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    img2 = cdf[img]

    plt.subplot(121), plt.imshow(img, 'gray'), plt.title('img')
    plt.subplot(122), plt.plot(cdf_normalized, color='b')
    plt.hist(img2.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256]), plt.ylim([0, 4000])
    plt.legend(('cdf', 'histogram'), loc = 'upper left')
    plt.show()

    #OpenCV 中的直方图均衡化函数为cv2.equalizeHist()。这个函数的输入图片仅仅是一副灰度图像，
    # 输出结果是直方图均衡化之后的图像。下边的代码还是对上边的那幅图像进行直方图均衡化：
    equ = cv.equalizeHist(img)
    res = np.hstack((img, equ))
    # stacking images side-by-side
    plt.imshow(res, 'gray'), plt.title('res')
    plt.show()

    #两幅图像中雕像的面图，由于太亮我们丢失了很多信息。

    #为了解决这个问题，我们需要使用自适应的直方图均衡化。这种情况下，整幅图像会被分成很多小块，
    # 这些小块被称为“tiles”（在OpenCV 中tiles 的大小默认是8x8），然后再对每一个小块分别进行直
    # 方图均衡化（跟前面类似）。所以在每一个的区域中，直方图会集中在某一个小的区域中（除非有噪声
    # 干扰）。如果有噪声的话，噪声会被放大。为了避免这种情况的出现要使用对比度限制。对于每个小块
    # 来说，如果直方图中的bin 超过对比度的上限的话，就把 其中的像素点均匀分散到其他bins 中，然后
    # 在进行直方图均衡化。最后，为了去除每一个小块之间“人造的”（由于算法造成）边界，再使用双线性
    # 差值，对小块进行缝合。下面的代码显示了如何使用OpenCV 中的CLAHE。
    img = cv.imread('tsukuba_l.png', 0)

    # create a CLAHE object (Arguments are optional).
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    plt.subplot(121), plt.imshow(img, 'gray')
    plt.subplot(122), plt.imshow(cl1, 'gray')
    plt.show()