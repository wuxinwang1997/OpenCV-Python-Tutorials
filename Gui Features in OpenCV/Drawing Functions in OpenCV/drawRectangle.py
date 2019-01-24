import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

if __name__ == "__main__":
    # Create a black image
    img = np.zeros((512, 512, 3), np.uint8)

    # Draw a rectangle
    cv.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
    cv.circle(img, (447, 63), 63, (0, 0, 255), -1)
    plt.imshow(img)
    plt.show()