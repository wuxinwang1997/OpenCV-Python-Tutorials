# -*- coding : utf-8 -*-
"""
@ author: wuxin
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":

    img = cv.imread('fly.jpg', 0)

    # Create SUFT object. You can specify params here or later.
    # Here I set Hessian Threshold to 400
    surf = cv.xfeatures2d.SURF_create(400)

    # Find kepoints and descriptors directly
    kp, des = surf.detectAndCompute(img, None)
    print(len(kp))

    # Check present Hessian threshold
    print(surf.getHessianThreshold())

    # We set it to sime 50000. Remember, it is just for represent in picture.
    # In actucal cases, it is better to have a value 300-500
    surf.setHessianThreshold(50000)

    # Again compute ketpoints and check its number.
    kp, des = surf.detectAndCompute(img, None)
    print(len(kp))

    img2 = cv.drawKeypoints(img, kp, None, (255, 0, 0), 4)
    plt.imshow(img2), plt.title('SURT')
    plt.xticks([]), plt.yticks([])
    plt.show()

    # Check upright flag, if it False, set it to True
    print(surf.getUpright())
    surf.setUpright(True)

    # Recompute the feature points and draw it
    kp = surf.detect(img, None)
    img3 = cv.drawKeypoints(img, kp, None, (255, 0, 0), 4)
    plt.imshow(img3), plt.title('U-SURT')
    plt.xticks([]), plt.yticks([])
    plt.show()

    #Find size of descriptor
    print(surf.descriptorSize())

    # That means flag, "extended" is False
    print(surf.getExtended())

    # So we make it to True to get 128-dim descriptors.
    surf.setExtended(True)
    kp, des = surf.detectAndCompute(img, None)
    print(surf.descriptorSize())
    print(des.shape)

