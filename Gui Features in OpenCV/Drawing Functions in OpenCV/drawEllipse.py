import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

if __name__ == "__main__":
    # Create a black image
    img = np.zeros((512, 512, 3), np.uint8)

    # Draw a ellipse
    cv.ellipse(img, (256, 256), (100, 50), 0, 0, 180, 255, -1)
    plt.imshow(img)
    plt.show()