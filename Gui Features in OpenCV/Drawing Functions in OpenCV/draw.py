import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

if __name__ == "__main__":
    # Create a black image
    img = np.zeros((512, 512, 3), np.uint8)

    # Draw a diagonal blue line with thickness of 5 px
    cv.line(img, (0, 0), (511, 511), (255, 0, 0), 5)

    # Draw a rectangle
    cv.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)

    # Draw a circle
    cv.circle(img, (447, 63), 63, (0, 0, 255), -1)

    # Draw a ellipse
    cv.ellipse(img, (256, 256), (100, 50), 0, 0, 180, 255, -1)

    # Draw a polygon
    pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    # 这里的 reshape 的第一个参数是-1，表明这一维的长度是根据后面的唯独计算出来的
    cv.polylines(img, [pts], True, (0, 255, 255), 2)

    # Print Text
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(img, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 2)
    # plt.imshow(img)
    # plt.show()
    winname = 'example'
    cv.namedWindow(winname)
    cv.imshow(winname, img)
    cv.waitKey(0)
    cv.destroyAllWindows(winname)
