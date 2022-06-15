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
start=True
while(start):
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
    avoid_mask = cv2.resize(avoid_mask, (640, 360))[220:270, :]
    avoid = np.expand_dims(avoid_mask, 2)
    avoid = np.concatenate((avoid, avoid, avoid), axis=2)
    cv2.putText(avoid, str(avoid_mask.sum()//255), (100, 30), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite("img/avoid"+str(cnt)+".jpg", avoid)
    
    #cv2.imshow("avoid_mask", avoid_mask)

    black_mask=np.all((seg_img == [0, 128, 0]), -1).astype(np.uint8) * 255
    black_mask = cv2.resize(black_mask, (640, 360))[250:320, :]
    black_line = np.expand_dims(black_mask, 2)
    black_line = np.concatenate((black_line, black_line, black_line), axis=2)
    cv2.putText(black_line, str(black_line.sum()//255), (100, 30), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite("img/black_line"+str(cnt)+".jpg", avoid)

    final_mask = np.all((seg_img == [128, 0, 0]), -1).astype(np.uint8) * 255
    final_mask = cv2.resize(final_mask, (640, 360))[200:300, :]
    final = np.expand_dims(final_mask, 2)
    final = np.concatenate((final, final, final), axis=2)
    #cv2.imshow("final_mask", final_mask)
    '''if black_mask.sum()>255*100 and direction!="Null":
            if direction=="right":
                robot.set_motors(0.09*2, 0.14*2)
                time.sleep(0.7)
                robot.stop()
                direction="Null"
                print("turn left")
            elif direction =="left":
                robot.set_motors(0.14*2, 0.09*2)
                time.sleep(0.7)
                robot.stop()
                direction="Null"
                print("turn right")'''
    if avoid_mask.sum() >= 255 * 3000:
        #print("avoid")

        if avoid_mask[:, :320].sum() > avoid_mask[:, 320:].sum():
            #setting 1
            #robot.set_motors(0.14*2, 0.11*2)
            #time.sleep(0.5)
            #setting 2
            #robot.right(0.005 * 70)
            #time.sleep(0.3)
            #robot.forward(0.2)
            #time.sleep(0.35)
            ##################
            #setting3
            robot.right(0.002 * 70)
            time.sleep(0.25)
            robot.stop()
            robot.set_motors(0.09*2, 0.14*2)
            time.sleep(0.65)
            robot.stop()
            robot.forward(0.2)
            time.sleep(0.1)
            robot.stop()
            direction="right"
            print("avoid turn right")
        else:
            #setting 1
            #robot.set_motors(0.11*2, 0.14*2)
            #time.sleep(0.5)
            #setting 2
            #robot.left(0.005 * 70)
            #time.sleep(0.3)
            #robot.forward(0.2)
            #time.sleep(0.35)
            ######################
            #setting3
            robot.left(0.002 * 70)
            time.sleep(0.25)
            robot.stop()
            robot.set_motors(0.11*2, 0.09*2)
            time.sleep(1.1)
            robot.stop()
            robot.forward(0.2)
            time.sleep(0.2)
            robot.stop()
            direction="left"
            print("avoid turn left")
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
                print("stop")
                #start=False
            else:
                robot.stop()
                print("stop")
                #start=False

        try:
            line_mean = int(np.mean([left[0][0], right[0][0]]))
            dist = 320 - line_mean

            cv2.circle(red_line, (320, 320), 10, (255, 215, 0), 2)
            cv2.circle(red_line, (left[0][0], 340), 10, (255, 0, 0), 2)
            cv2.circle(red_line, (right[0][0], 340), 10, (255, 0, 0), 2)
            cv2.circle(red_line, (line_mean, 340), 10, (255, 0, 0), 2)
            cv2.putText(red_line, str(int(dist)), (100, 30), cv2.FONT_HERSHEY_SIMPLEX,1,  (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite("img/red_line"+str(cnt)+".jpg", red_line)
            #cv2.imshow("red_line", red_line)

            if dist >= -70 and dist <= 70:
                robot.forward(0.2*(3000-(avoid_mask.sum()/255))/3000)
                time.sleep(0.3)
                robot.stop()
                print("forward")
            elif dist > 70:
                robot.left(0.0012 * (dist/1.1))
                time.sleep(0.25)
                robot.stop()
                print("follow red and turn left")
            else:
                robot.right(0.0012 * -(dist/1.1))
                time.sleep(0.25)
                robot.stop()
                print("follow red and turn right")
        except:
            pass



#camera = Camera()
#camera.observe(execute)
