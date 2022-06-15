from PathPlanning.planner import Planner
import PathPlanning.utils as utils
import cv2
import sys
sys.path.append("..")


class PlannerAStar(Planner):
    def __init__(self, m, inter=10):
        super().__init__(m)
        self.inter = inter
        self.initialize()

    def initialize(self):
        self.queue = []
        self.parent = {}
        self.h = {}  # Distance from node to goal
        self.g = {}  # Distance from start to node
        self.goal_node = None

    def planning(self, start=(100, 200), goal=(375, 520), inter=None, img=None):
        if inter is None:
            inter = self.inter
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        # Initialize
        self.initialize()
        self.queue.append(start)
        self.parent[start] = None
        self.g[start] = 0
        self.h[start] = utils.distance(start, goal)
        while(1):
            # TODO: A Star Algorithm

            min_f = 99999
            min_id = -1

            # get the node with the smallest f
            for i, node in enumerate(self.queue):
                # f = g + h
                f = self.g[node] + self.h[node]
                if f < min_f:
                    min_f = f
                    min_id = i

            # pop the node with the smallest f
            p = self.queue.pop(min_id)

            # meet obstacle
            if self.map[p[1], p[0]] < 0.5:
                continue

            # Check if the goal is reached
            if utils.distance(p, goal) < inter:
                self.goal_node = p
                break

            # generate p's 8 successors
            pts_next = [(p[0]+inter, p[1]), (p[0]-inter, p[1]), (p[0], p[1]+inter), (p[0], p[1]-inter),
                        (p[0]+inter, p[1]+inter), (p[0]-inter, p[1]-inter), (p[0]+inter, p[1]-inter), (p[0]-inter, p[1]+inter)]

            for p_next in pts_next:
                # if the successor is not in parent
                if p_next not in self.parent:
                    self.queue.append(p_next)
                    # store next point's parent is current p
                    self.parent[p_next] = p
                    # estimation of next point g(v)
                    self.g[p_next] = self.g[p] + inter
                    # update h
                    self.h[p_next] = utils.distance(p_next, goal)

                # if the successor is in parent
                elif self.g[p_next] > self.g[p] + inter:
                    self.parent[p_next] = p
                    self.g[p_next] = self.g[p] + inter

            if img is not None:
                cv2.circle(img, (start[0], start[1]), 5, (0, 0, 1), 3)
                cv2.circle(img, (goal[0], goal[1]), 5, (0, 1, 0), 3)
                cv2.circle(img, p, 2, (0, 0, 1), 1)
                img_ = cv2.flip(img, 0)
                cv2.imshow("A* Test", img_)
                k = cv2.waitKey(1)
                if k == 27:
                    break

        # Extract path
        path = []
        p = self.goal_node
        if p is None:
            return path
        while(True):
            path.insert(0, p)
            if self.parent[p] is None:
                break
            p = self.parent[p]
        if path[-1] != goal:
            path.append(goal)
        return path
