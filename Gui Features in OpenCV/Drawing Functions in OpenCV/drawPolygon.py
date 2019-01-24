import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

if __name__ == "__main__":
    # Create a black image
    img = np.zeros((512, 512, 3), np.uint8)

    # Draw a polygon
    pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    # 这里的 reshape 的第一个参数是-1，表明这一维的长度是根据后面的唯独计算出来的
    cv.polylines(img, [pts], True, (0, 255, 255), 2)
    plt.imshow(img)
    plt.show()