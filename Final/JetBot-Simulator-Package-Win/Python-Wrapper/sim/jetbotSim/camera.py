import sys
sys.path.append('./jetbotSim')
import numpy as np
import cv2
import websocket
from websocket import create_connection
import threading
import time
import config

class Camera():
    def __init__(self):    
        self.ws = None
        self.wst = None
        self._connect_server(config.ip, config.actor)
        self.buffer = None
        self.on_change = False
        self.old_value = None

    def _connect_server(self, ip, actor):
        self.ws = websocket.WebSocketApp("ws://%s/%s/camera/subscribe"%(ip, actor), on_message = lambda ws,msg: self._on_message_camera(ws, msg))
        self.wst = threading.Thread(target=self.ws.run_forever)
        self.wst.daemon = True
        self.wst.start()
        time.sleep(1)   #wait for connect
    
    def _on_message_camera(self, ws, msg):
        self.buffer = msg
        self.on_change = True
        
    def observe(self, execute):
        print("\n[Start Observation]")
        while True:
            if self.buffer is not None and self.on_change:
                nparr = np.fromstring(self.buffer, np.uint8)
                value = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                execute({"new":value.copy(), "old":self.old_value})
                self.old_value = value.copy()
                self.on_change = False
            k = cv2.waitKey(1)
            if k == 27:
                print("\n[End Observation]")
                break

# Observation Test
if __name__ == "__main__":
    def execute(change):
        global ts
        img_re = cv2.resize(change["new"], (640,360))
        print("\rReceive Frame", ts, end="")
        ts += 1
        cv2.imshow("test", img_re)

    ts = 0
    camera = Camera()
    camera.observe(execute)
        