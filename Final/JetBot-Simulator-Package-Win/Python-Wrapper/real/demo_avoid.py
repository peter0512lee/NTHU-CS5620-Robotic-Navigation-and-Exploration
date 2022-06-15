from jetbot import Robot, Camera
from inference import segmentation

import numpy as np
import cv2
import time
camera = Camera.instance(width=960, height=540, capture_width=1280, capture_height=720)
#import matplotlib.pyplot as plt
robot = Robot()
direction="Null"
cnt = 0
while(True):
    cnt+=1
    # Visualize
    img = camera.value
    img = cv2.resize(img, (256, 256))
    #img = cv2.resize(change["new"], (256, 256))
    print(cnt)
    cv2.imwrite("img/camera_"+str(cnt)+".jpg", img)
    #cv2.imshow("aaa", img)
    #cv2.waitKey(1)
    # Segmentation
    seg_img = segmentation(img)
    cv2.imwrite("img/segmentation_"+str(cnt)+".jpg", seg_img)

    mask = np.all((seg_img == [0, 0, 128]), -1).astype(np.uint8) * 255
    mask = cv2.resize(mask, (640, 360))
    red_line = np.expand_dims(mask, 2)
    red_line = np.concatenate((red_line, red_line, red_line), axis=2)

    avoid_mask = np.all((seg_img == [0, 128, 128]), -1).astype(np.uint8) * 255
    avoid_mask = cv2.resize(avoid_mask, (640, 360))[170:270, :]
    avoid = np.expand_dims(avoid_mask, 2)
    avoid = np.concatenate((avoid, avoid, avoid), axis=2)
    #cv2.imshow("avoid_mask", avoid_mask)

    final_mask = np.all((seg_img == [128, 0, 0]), -1).astype(np.uint8) * 255
    final_mask = cv2.resize(final_mask, (640, 360))[200:300, :]
    final = np.expand_dims(final_mask, 2)
    final = np.concatenate((final, final, final), axis=2)
    #cv2.imshow("final_mask", final_mask)

    if avoid_mask.sum() > 255 * 500:
        print("avoid")

        if avoid_mask[:, :320].sum() > avoid_mask[:, 320:].sum():
            robot.set_motors(0.14*1.5, 0.09*1.5)
            time.sleep(0.5)
            robot.stop()
            direction="right"
        else:
            robot.set_motors(0.09*1.5, 0.11*1.5)
            time.sleep(0.5)
            robot.stop()
            direction="left"
    else:

        try:
            coord = cv2.findNonZero(mask[300:360, :])
            left = np.min(coord, axis=0) + (0, 300)
            right = np.max(coord, axis=0) + (0, 300)
        except:
            if final_mask.sum() > 255 * 15:
                robot.forward(0.2)
                time.sleep(0.5)
                robot.stop()
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

            if dist >= -80 and dist <= 80:
                robot.forward(0.2)
                time.sleep(0.5)
                robot.stop()
            elif dist > 80:
                robot.left(0.001 * dist)
                time.sleep(0.2)
                robot.stop()
            else:
                robot.right(0.001 * -dist)
                time.sleep(0.2)
                robot.stop()
        except:
            pass



#camera = Camera()
#camera.observe(execute)
