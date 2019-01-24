# -*-cofing:utf-8 -*-
"""
@ authot: wuxin
"""

import cv2 as cv
import numpy as np

if __name__ == '__main__':

    cap = cv.VideoCapture('vtest.avi')

    # params for ShiTomas corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15,15),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TermCriteria_COUNT, 10, 0.03))
    # Create some radom colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('Lucas-Kanade Optical Flow.avi', fourcc, 20.0, (old_frame.shape[1], old_frame.shape[0]))

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while(1):
        ret, frame = cap.read()
        if ret == True:
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # calculate optical flow
            p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                frame = cv.circle(frame, (a, b), 5, color[i].tolist(), -1)
            img = cv.add(frame, mask)

            out.write(img)
            cv.imshow('Lucas-Kanade Optical Flow', img)
            k = cv.waitKey(60) & 0xff
            if k == 27:
                break
        else:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cap.release()
    out.release()
    cv.destroyAllWindows()
