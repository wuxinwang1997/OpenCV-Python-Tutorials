import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

if __name__ == "__main__":
    # Create a black image
    img = np.zeros((512, 512, 3), np.uint8)

    # Draw a diagonal blue line with thickness of 5 px
    cv.line(img, (0, 0), (511, 511), (255, 0, 0), 5)
    plt.imshow(img)
    plt.show()
