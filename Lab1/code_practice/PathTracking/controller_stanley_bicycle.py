from PathTracking.controller import Controller
import PathTracking.utils as utils
import sys
import numpy as np
sys.path.append("..")


class ControllerStanleyBicycle(Controller):
    def __init__(self, kp=0.5):
        self.path = None
        self.kp = kp

    # State: [x, y, yaw, delta, v, l]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None, None

        # Extract State
        x, y, yaw, delta, v, l = info["x"], info["y"], info["yaw"], info["delta"], info["v"], info["l"]

        # Search Front Wheel Target
        front_x = x + l*np.cos(np.deg2rad(yaw))
        front_y = y + l*np.sin(np.deg2rad(yaw))
        vf = v / np.cos(np.deg2rad(delta))
        min_idx, min_dist = utils.search_nearest(self.path, (front_x, front_y))
        target = self.path[min_idx]

        # TODO: Stanley Control for Bicycle Kinematic Model
        theta_p = target[2]
        theta_e = (theta_p - yaw) % 360
        if theta_e > 180:
            theta_e = theta_e - 360
        e = np.dot([front_x - target[0], front_y - target[1]],
                   [np.cos(np.deg2rad(theta_p + 90)), np.sin(np.deg2rad(theta_p + 90))])
        next_delta = np.rad2deg(np.arctan2(-self.kp*e, vf)) + theta_e
        return next_delta, target
