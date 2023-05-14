import numpy as np
from math import sin, cos, pi, sqrt

import pygame
from gymnasium import spaces
import numpy as np
from pygame.locals import *


class SimplePlayer:
    action_space = spaces.Box(
        np.array([-1, -1]).astype(np.float32),
        np.array([+1, +1]).astype(np.float32),
        shape=(2,)
    )  # steer, gas, brake

    def __init__(self):
        # Physics constants
        self.gravity = 0.08
        self.thruster_amplitude = 0.04
        self.diff_amplitude = 0.003
        self.thruster_mean = 0.04
        self.mass = 1
        self.inertia = 1
        self.arm = 25
        self.friction = 0.98
        # Movement vars
        (self.a, self.ad, self.add) = (0, 0, 0)
        (self.x, self.xd, self.xdd) = (0, 0, 0)
        (self.y, self.yd, self.ydd) = (0, 0, 0)

        self.targets = []
        self.current_target_id = 0
        self.target_counter = 0

    def reset_player(self, targets, x, y):
        (self.a, self.ad, self.add) = (0, 0, 0)
        (self.x, self.xd, self.xdd) = (x, 0, 0)
        (self.y, self.yd, self.ydd) = (y, 0, 0)

        self.targets = targets
        self.current_target_id = 0
        self.target_counter = 0

    @staticmethod
    def unit_vector(vector):
        return vector / np.linalg.norm(vector)

    def angle_target_and_velocity(self):
        xt, yt = self.current_target_xy()
        return (np.arctan2(yt - self.y, xt - self.x) - np.arctan2(self.yd, self.xd)) / np.pi

    def angle_to_target(self):
        xt, yt = self.current_target_xy()
        return np.arctan2(yt - self.y, xt - self.x) / np.pi

    def dist_to_target(self):
        target_x, target_y = self.current_target_xy()
        return sqrt((self.x - target_x) ** 2 + (self.y - target_y) ** 2)

    def current_target_xy(self):
        return self.targets[self.current_target_id]

    def next_target_xy(self):
        return self.targets[self.current_target_id + 1]

    def set_next_target(self):
        self.current_target_id += 1
        self.target_counter += 1

    def update(self, action):
        (action0, action1) = (action[0], action[1])

        # # Initialize accelerations
        # thruster_left = self.thruster_mean
        # thruster_right = self.thruster_mean
        #
        # thruster_left += action0 * self.thruster_amplitude
        # thruster_right += action0 * self.thruster_amplitude
        # thruster_left += action1 * self.diff_amplitude
        # thruster_right -= action1 * self.diff_amplitude
        #
        # # Calculating accelerations with Newton's laws of motions
        # self.xdd = -(thruster_left + thruster_right) * sin(self.a * pi / 180) / self.mass
        # self.ydd = self.gravity - (thruster_left + thruster_right) * cos(self.a * pi / 180) / self.mass
        # self.add = self.arm * (thruster_right - thruster_left) / self.inertia

        # self.xdd -= self.friction * self.xd
        # self.ydd -= self.friction * self.yd
        self.xd = self.xd * self.friction + action1
        self.yd = self.yd * self.friction - action0

        # print(f'{action} | {self.add} {self.xdd} {self.ydd}')

    def move(self, dt):
        # self.xd += self.xdd
        # self.yd += self.ydd
        # self.ad += self.add
        # self.x += self.xd
        # self.y += self.yd
        # self.a += self.ad
        self.xd += self.xdd * dt
        self.yd += self.ydd * dt
        self.ad += self.add * dt
        self.x += self.xd * dt
        self.y += self.yd * dt
        self.a += self.ad * dt


class SimpleHumanPlayer(SimplePlayer):
    def __init__(self):
        self.name = "Human"
        self.alpha = 255
        super().__init__()

    def act(self, obs):
        action0 = 0
        action1 = 0
        pressed_keys = pygame.key.get_pressed()
        if pressed_keys[K_UP]:
            action0 += 1
        elif pressed_keys[K_DOWN]:
            action0 -= 1
        if pressed_keys[K_LEFT]:
            action1 -= 1
        elif pressed_keys[K_RIGHT]:
            action1 += 1
        return action0, action1
