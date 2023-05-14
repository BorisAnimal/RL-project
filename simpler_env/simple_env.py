"""
Inspired by https://github.com/thowell/achtung/blob/main/achtung.py
and https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/box2d/car_racing.py
"""
from typing import Optional

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import EzPickle
from pygame import Rect

from simple_player import SimpleHumanPlayer

pygame.init()

N_TARGETS = 1000

STATE_W = 100
STATE_H = 100
STATE_BORDER = 10
TARGET_RADIUS = 2

VIDEO_W = 600
VIDEO_H = 600
WINDOW_W = 800
WINDOW_H = 800

FPS = 30


class SimpleEnv(gym.Env, EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "state_pixels",
        ],
        "render_fps": FPS,
    }

    def __init__(self,
                 player,
                 time_limit=60,
                 render_mode: Optional[str] = None,
                 verbose: bool = False,
                 lap_complete_percent: float = 0.95,
                 domain_randomize: bool = False,
                 continuous: bool = True,
                 ):
        EzPickle.__init__(
            self,
            render_mode,
            verbose,
            lap_complete_percent,
            domain_randomize,
            continuous,
        )
        # pygame
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.fps_clock = None
        self.myfont = None
        self.render_mode = render_mode
        if self.render_mode == 'human':
            pygame.init()
            self.window = pygame.display.set_mode((WINDOW_W, WINDOW_H))

            pygame.display.init()
            pygame.display.set_caption('Drone environments')

            pygame.font.init()
            self.myfont = pygame.font.SysFont("Comic Sans MS", 20)
            self.fps_clock = pygame.time.Clock()

        self.window_width = WINDOW_W
        self.window_height = WINDOW_H

        # Initialize game variables
        self.FPS = FPS
        self.dt = 1 / self.FPS
        self.time = 0
        self.time_limit = time_limit
        self.game_over = False
        self.frame_count = 0

        # colors
        self.target_color = np.array([255, 100, 100])
        self.next_target_color = np.array([100, 100, 255])
        self.player_color = np.array([100, 255, 100])
        self.background_color = np.array([255, 255, 255])
        self.restricted_background_color = np.array([50, 50, 50])

        self.static_canvas = self.static_background()

        # Init agents
        self.player = player
        self.rays8_mask = STATE_BORDER / 2 * np.array(
            [(np.cos(np.pi * i / 4), np.sin(np.pi * i / 4)) for i in range(8)])
        # self.bots = bots

        # gym
        self.action_space = self.player.action_space
        self.observation_space = spaces.Box(
            low=np.array([-STATE_W, -STATE_H, -STATE_W, -STATE_H, -STATE_W, -STATE_H, 0, 0, 0, 0, 0, 0, 0, 0]).astype(
                np.float32),
            high=np.array([STATE_W, STATE_H, STATE_W, STATE_H, STATE_W, STATE_H, 1, 1, 1, 1, 1, 1, 1, 1]).astype(
                np.float32),
        )  # [dx, dy, dx_target, dy_target] + rays8
        self.reset()

    def static_background(self):
        canvas = pygame.Surface((STATE_W, STATE_H))
        canvas.fill(self.restricted_background_color)
        canvas.fill(
            self.background_color,
            rect=Rect(STATE_BORDER, STATE_BORDER, STATE_W - 2 * STATE_BORDER, STATE_H - 2 * STATE_BORDER)
        )
        return canvas

    def check_point_restricted(self, x, y):
        if x < 0 or x > STATE_W or y < 0 or y > STATE_H:
            return True
        return np.allclose(self.static_canvas.get_at((int(x), int(y)))[:3], self.restricted_background_color)

    def get_obs(self):
        player = self.player
        x, y = player.x, player.y
        xt, yt = player.current_target_xy()
        xt2, yt2 = player.next_target_xy()

        xd, yd = player.xd, player.yd
        dx_target = xt - player.x
        dy_target = yt - player.y
        dx_target2 = xt2 - player.x
        dy_target2 = yt2 - player.y
        # dist = player.dist_to_target()
        # angle = player.angle_to_target()
        # angle_target_and_velocity = player.angle_target_and_velocity()
        rays8 = [1 if self.check_point_restricted(*point) else 0 for point in np.array([x, y]) + self.rays8_mask]

        return np.array(
            [xd, yd, dx_target, dy_target, dx_target2, dy_target2] + rays8
        ).astype(np.float32)

    def check_round_over(self):
        self.game_over = self.time >= self.time_limit or \
                         self.player.x < 0 or self.player.x > STATE_W or \
                         self.player.y < 0 or self.player.y > STATE_H
        return self.game_over

    def reward(self):
        dist = self.player.dist_to_target()
        reward = 0
        # 0. Reward per step survived
        # reward += 1 / self.FPS
        # 1. Penalty according to the distance to target
        reward -= dist / (100 * self.FPS)
        # 2. Reward if close to target
        if dist <= TARGET_RADIUS:
            self.player.set_next_target()
            reward += 100
        # 3. Penalty if out of playground
        if self.player.x < 0 or self.player.x > STATE_W or self.player.y < 0 or self.player.y > STATE_H:
            reward -= 1000
        # 4. Penalty if too high speed
        if abs(self.player.ad) > 5 or abs(self.player.xd) > 40 or abs(self.player.yd) > 40:
            reward -= 10
        # 5. penalty for high angle
        if abs(self.player.a) % 180 > 30:
            reward -= 10
        # 6. penalty if too close to border
        if self.player.x < STATE_BORDER or self.player.y < STATE_BORDER or self.player.x > STATE_W - STATE_BORDER or self.player.y > STATE_H - STATE_BORDER:
            reward -= 1
        return reward

    def step(self, action):
        self.time += self.dt
        # update current player
        self.player.update(action)
        # TODO: handle bots
        if not self.game_over:
            self.apply_to_all_agents(lambda agent: agent.move(self.dt))
            # check round over
            self.check_round_over()

        state = self.get_obs()
        reward = self.reward()
        done = self.game_over
        return state, reward, done, False, {}

    def reset(self, **kwargs):
        self.game_over = False
        self.time = 0
        self.frame_count = 0

        targets = np.hstack((
            np.random.randint(STATE_BORDER, STATE_W - STATE_BORDER, (N_TARGETS, 1)),
            np.random.randint(STATE_BORDER, STATE_H - STATE_BORDER, (N_TARGETS, 1))
        ))
        x, y = int(STATE_W / 2), int(STATE_H / 2)
        self.apply_to_all_agents(lambda agent: agent.reset_player(targets=targets, x=x, y=y))

        return self.get_obs(), {}

    def render_actor_and_target(self, canvas, actor):
        # Target
        target_center = actor.current_target_xy()
        pygame.draw.circle(canvas, self.target_color, target_center, TARGET_RADIUS)

        target2_center = actor.next_target_xy()
        pygame.draw.circle(canvas, self.next_target_color, target2_center, TARGET_RADIUS)

        # Actor
        actor_xy = (actor.x, actor.y)
        # actor_asset_rotated = pygame.transform.rotate(actor.player_asset, actor.a)
        pygame.draw.rect(canvas, self.player_color, (actor_xy, (1, 1)), 1)

    def apply_to_all_agents(self, func):
        func(self.player)
        # for bot in self.bots:
        #     func(bot)

    def render(self, mode=None):
        canvas = self.static_canvas.copy()
        self.apply_to_all_agents(lambda agent: self.render_actor_and_target(canvas, agent))

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(pygame.transform.scale(canvas, (WINDOW_W, WINDOW_H)), (0, 0))

            textsurface = self.myfont.render(f"Collected: {self.player.target_counter}", False, (10, 10, 10))
            self.window.blit(textsurface, (20, 20))
            textsurface3 = self.myfont.render(f"Time: {int(self.time)}", False, (10, 10, 10))
            self.window.blit(textsurface3, (20, 50))

            pygame.event.pump()
            pygame.display.update()

            self.fps_clock.tick(self.FPS)
        self.frame_count += 1
        return np.array(pygame.surfarray.array3d(canvas), dtype=np.uint8)


if __name__ == '__main__':
    player = SimpleHumanPlayer()
    env = SimpleEnv(player=player, render_mode='human')
    # env = EnvSacCnn(player=player, render_mode=None)
    env.render()
    done = False

    # game
    while not done:
        action = player.act(123)
        # step
        obs, reward, done, terminated, info = env.step(action)
        env.render()

        print(f'{reward}')

    print(f"Terminated at {env.frame_count} frame")
