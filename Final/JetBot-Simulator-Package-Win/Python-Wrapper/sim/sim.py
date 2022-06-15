from jetbotSim import Robot, Camera
import numpy as np
import cv2


def execute(change):
    global robot

    img = cv2.resize(change["new"], (640, 360))

    # Visualize
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([37, 148, 58])
    upper_red = np.array([255, 255, 236])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(img, img, mask=mask)

    coord = cv2.findNonZero(mask)
    left = np.min(coord, axis=0)
    right = np.max(coord, axis=0)
    try:
        line_mean = int(np.mean([left[0][0], right[0][0]]))
        dist = 320 - line_mean

        cv2.circle(res, (320, 320), 10, (255, 215, 0), 2)
        cv2.circle(res, (left[0][0], 340), 10, (255, 255, 255), 2)
        cv2.circle(res, (right[0][0], 340), 10, (255, 255, 255), 2)
        cv2.circle(res, (line_mean, 340), 10, (255, 255, 255), 2)
        cv2.imshow("res", res)

        if dist >= -20 and dist <= 20:
            robot.forward(0.2)
        elif dist > 20:
            robot.left(0.002 * dist)
        else:
            robot.right(0.002 * -dist)

    except:
        robot.stop()

    cv2.imshow("camera", img)
    cv2.imshow('res', res)


robot = Robot()
camera = Camera()
camera.observe(execute)
