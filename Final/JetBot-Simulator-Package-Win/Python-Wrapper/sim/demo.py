from jetbotSim import Robot, Camera
from segmentation.inference import segmentation

import numpy as np
import cv2
import matplotlib.pyplot as plt


def execute(change):
    global robot

    # Visualize
    img = cv2.resize(change["new"], (256, 256))
    cv2.imshow("camera", img)

    # Segmentation
    seg_img = segmentation(img)
    cv2.imshow("segmentation", seg_img)

    mask = np.all((seg_img == [0, 0, 128]), -1).astype(np.uint8) * 255
    mask = cv2.resize(mask, (640, 360))
    red_line = np.expand_dims(mask, 2)
    red_line = np.concatenate((red_line, red_line, red_line), axis=2)

    try:
        coord = cv2.findNonZero(mask[300:360, :])
        left = np.min(coord, axis=0) + (0, 300)
        right = np.max(coord, axis=0) + (0, 300)
    except:
        robot.stop()

    try:
        line_mean = int(np.mean([left[0][0], right[0][0]]))
        dist = 320 - line_mean

        cv2.circle(red_line, (320, 320), 10, (255, 215, 0), 2)
        cv2.circle(red_line, (left[0][0], 340), 10, (255, 0, 0), 2)
        cv2.circle(red_line, (right[0][0], 340), 10, (255, 0, 0), 2)
        cv2.circle(red_line, (line_mean, 340), 10, (255, 0, 0), 2)

        cv2.imshow("red_line", red_line)

        if dist >= -20 and dist <= 20:
            robot.forward(0.5)
        elif dist > 20:
            robot.left(0.0002 * dist)
        else:
            robot.right(0.0002 * -dist)
    except:
        robot.stop()


robot = Robot()
camera = Camera()
camera.observe(execute)
