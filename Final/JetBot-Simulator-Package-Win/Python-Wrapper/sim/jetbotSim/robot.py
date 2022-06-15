import sys
sys.path.append('./jetbotSim')
import numpy as np
import cv2
import websocket
from websocket import create_connection
import threading
import time
import cv2
import numpy as np
import struct
import json
import config

class Robot():
    def __init__(self):    
        self.ws = None
        self._connect_server(config.ip, config.actor)
        self._left_motor = 0
        self._right_motor = 0
        self.reset()

    def _connect_server(self, ip, actor):
        self.ws = create_connection("ws://%s/%s/controller/session"%(ip, actor))
        time.sleep(1)   #wait for connect
    
    def _move_to_wheel(self, value):
        length = 2 * np.pi * config.wheel_rad 
        angular_vel = 360 * (1000*value / length)
        return angular_vel

    # Control Command
    def set_left_motor(self, value):
        left_ang = self._move_to_wheel(value)
        jsonStr = json.dumps({'leftMotor':left_ang, 'rightMotor':0.0, 'flag':1})
        self.ws.send(jsonStr)

    def set_right_motor(self, value):
        right_ang = self._move_to_wheel(value)
        jsonStr = json.dumps({'leftMotor':0.0, 'rightMotor':right_ang, 'flag':2})
        self.ws.send(jsonStr)
    
    def set_motor(self, value_l, value_r):
        left_ang = self._move_to_wheel(value_l)
        right_ang = self._move_to_wheel(value_r)
        jsonStr = json.dumps({'leftMotor':left_ang, 'rightMotor':right_ang, 'flag':4})
        self.ws.send(jsonStr)
    
    def add_motor(self, value_l, value_r):
        left_ang = self._move_to_wheel(value_l)
        right_ang = self._move_to_wheel(value_r)
        jsonStr = json.dumps({'leftMotor':left_ang, 'rightMotor':right_ang, 'flag':3})
        self.ws.send(jsonStr)

    def forward(self, value):
        ang = self._move_to_wheel(value)
        jsonStr = json.dumps({'leftMotor':ang, 'rightMotor':ang, 'flag':4})
        self.ws.send(jsonStr)
    
    def backward(self, value):
        ang = self._move_to_wheel(value)
        jsonStr = json.dumps({'leftMotor':-ang, 'rightMotor':-ang, 'flag':4})
        self.ws.send(jsonStr)

    def left(self, value):
        ang = self._move_to_wheel(value)
        jsonStr = json.dumps({'leftMotor':-ang, 'rightMotor':ang, 'flag':4})
        self.ws.send(jsonStr)

    def right(self, value):
        ang = self._move_to_wheel(value)
        jsonStr = json.dumps({'leftMotor':ang, 'rightMotor':-ang, 'flag':4})
        self.ws.send(jsonStr)

    def stop(self):
        jsonStr = json.dumps({'leftMotor':0.0, 'rightMotor':0.0, 'flag':4})
        self.ws.send(jsonStr)
    
    def reset(self):
        jsonStr = json.dumps({'leftMotor':0.0, 'rightMotor':0.0, 'flag':0})
        self.ws.send(jsonStr)

# Observation Test
if __name__ == "__main__":
    robot = Robot()
    robot.reset()
    time.sleep(1)

    robot.forward(0.2)
    time.sleep(1)
    robot.stop()
    time.sleep(1)

    robot.backward(0.2)
    time.sleep(1)
    robot.stop()
    time.sleep(1)
    
    robot.right(0.2)
    time.sleep(1)
    robot.stop()
    time.sleep(1)
    
    robot.left(0.2)
    time.sleep(1)
    robot.stop()
    time.sleep(1)
    
    robot.forward(0.2)
    time.sleep(1)
    robot.add_motor(-0.4,-0.2)
    time.sleep(1)
    robot.stop()