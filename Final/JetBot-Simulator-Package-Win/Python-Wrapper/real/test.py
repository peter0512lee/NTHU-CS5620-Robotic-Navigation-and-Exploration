from jetbot import Robot, Camera
import numpy as np
import cv2
from inference import segmentation
camera = Camera.instance(width=960, height=540, capture_width=1280, capture_height=720)

cnt = 0
#while(True):
def execute(cheang):
    global cnt
    cnt+=1
    # Visualize
    img = camera.value
    #print(img.dtype)
    #img = cv2.resize(img, (256, 256))
    #img = cv2.resize(change["new"], (256, 256))
    
    if(cnt%120 == 0):
        print(cnt)
        cv2.imwrite("train_img2/camera_"+str(cnt)+".jpg", img)
    #cv2.imshow("aaa", img)
        seg_img = segmentation(img)
        cv2.imwrite("train_img2/seg_"+str(cnt)+".jpg", seg_img)
    #mask = np.all((seg_img == [0, 0, 128]), -1).astype(np.uint8) * 255
    #mask = cv2.resize(mask, (640, 360))
    #red_line = np.expand_dims(mask, 2)
    #red_line = np.concatenate((red_line, red_line, red_line), axis=2)
    #coord = cv2.findNonZero(mask[300:360, :])
    #left = np.min(coord, axis=0) + (0, 300)
    #right = np.max(coord, axis=0) + (0, 300)
    #line_mean = int(np.mean([left[0][0], right[0][0]]))
    #print("line_mean:",line_mean)
    #dist = 320 - line_mean
    #print("left:",left[0][0])
    #print("right:",right[0][0])
   # print("dist:",dist)
    #cv2.circle(red_line, (320, 320), 10, (255, 215, 0), 2)
   # cv2.circle(red_line, (left[0][0], 340), 10, (255, 0, 0), 2)
   # cv2.circle(red_line, (right[0][0], 340), 10, (255, 0, 0), 2)
    #cv2.circle(red_line, (line_mean, 340), 10, (255, 0, 0), 2)
   # cv2.imwrite("img/seg_"+str(cnt)+".jpg", red_line)
    #cv2.imshow("bbb", seg_img)
    #cv2.waitKey(1)
robot = Robot()
#camera = Camera()
camera.observe(execute)
