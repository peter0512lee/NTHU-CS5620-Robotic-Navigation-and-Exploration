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

    avoid_mask = np.all((seg_img == [0, 128, 128]), -1).astype(np.uint8) * 255
    avoid_mask = cv2.resize(avoid_mask, (640, 360))[170:270, :]
    avoid = np.expand_dims(avoid_mask, 2)
    avoid = np.concatenate((avoid, avoid, avoid), axis=2)
    cv2.imshow("avoid_mask", avoid_mask)

    final_mask = np.all((seg_img == [128, 0, 0]), -1).astype(np.uint8) * 255
    final_mask = cv2.resize(final_mask, (640, 360))[200:300, :]
    final = np.expand_dims(final_mask, 2)
    final = np.concatenate((final, final, final), axis=2)
    cv2.imshow("final_mask", final_mask)

    if avoid_mask.sum() > 255 * 500:

        if avoid_mask[:, :320].sum() > avoid_mask[:, 320:].sum():
            robot.set_motor(0.25, 0.18)
        else:
            robot.set_motor(0.18, 0.25)
    else:

        try:
            coord = cv2.findNonZero(mask[300:360, :])
            left = np.min(coord, axis=0) + (0, 300)
            right = np.max(coord, axis=0) + (0, 300)
        except:
            if final_mask.sum() > 255 * 15:
                robot.forward(0.3)
            else:
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
                robot.forward(0.2)
            elif dist > 20:
                robot.left(0.0002 * dist)
            else:
                robot.right(0.0002 * -dist)
        except:
            pass


robot = Robot()
camera = Camera()
camera.observe(execute)
