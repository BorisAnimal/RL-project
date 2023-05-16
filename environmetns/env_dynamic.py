from typing import Optional

import numpy as np
import pygame

from environmetns.env_simple import SimpleEnv, STATE_H, STATE_W, STATE_BORDER
from environmetns.player import SimpleHumanPlayer, SimplePlayer

BIG_RADIUS = 12
SMALL_RADIUS = 8
INTER_POINT_MIN_DELTA = 15
INTER_POINT_MAX_DELTA = 30
CORRIDOR_POINTS = 20


class DynamicEnv(SimpleEnv):

    def __init__(self, player: SimplePlayer, time_limit=60, render_mode: Optional[str] = None):
        super().__init__(player, time_limit, render_mode)
        # if this value != player.target_count, on step call background will be redrawn
        self.last_render_target_counter = -1

    def reset(self, **kwargs):
        self.last_render_target_counter = -1
        return SimpleEnv.reset(self, **kwargs)

    def generate_next_target(self):
        self.background_cache = SimpleEnv.static_background(self)
        return SimpleEnv.generate_next_target(self)

    def step(self, action):
        if self.last_render_target_counter != self.player.target_counter:
            self.generate_restricted_zones()
            self.last_render_target_counter = self.player.target_counter
        return SimpleEnv.step(self, action)

    def generate_intermediate_point(self):
        p_x, p_y = self.player.current_xy()
        t_x, t_y = self.player.current_target_xy()

        x = (p_x + t_x) / 2
        y = (p_y + t_y) / 2

        x += np.random.randint(INTER_POINT_MIN_DELTA, INTER_POINT_MAX_DELTA) * 1 if np.random.random() < 0.5 else -1
        y += np.random.randint(INTER_POINT_MIN_DELTA, INTER_POINT_MAX_DELTA) * 1 if np.random.random() < 0.5 else -1
        if not self.check_point_restricted(x, y):
            return x, y
        return self.generate_intermediate_point()

    def generate_restricted_zones(self):
        self.background_cache = SimpleEnv.static_background(self)
        intermediate_point = self.generate_intermediate_point()
        canvas = self.background_cache  # parent's static_background (just borders)
        canvas.fill(self.restricted_background_color, (0, 0, STATE_W, STATE_H))
        # safe zone around player
        pygame.draw.circle(canvas, self.background_color, self.player.current_xy(), BIG_RADIUS)
        # safe zone around target
        pygame.draw.circle(canvas, self.background_color, self.player.current_target_xy(), BIG_RADIUS)
        # safe zone around intermediate point
        pygame.draw.circle(canvas, self.background_color, intermediate_point, BIG_RADIUS)
        # safe corridor between points
        p_x, p_y = self.player.current_xy()
        i_x, i_y = intermediate_point
        t_x, t_y = self.player.current_target_xy()
        for i in range(CORRIDOR_POINTS):
            coef = i / CORRIDOR_POINTS
            x = p_x * (1 - coef) + i_x * coef
            y = p_y * (1 - coef) + i_y * coef
            pygame.draw.circle(canvas, self.background_color, (x, y), SMALL_RADIUS)
        for i in range(CORRIDOR_POINTS):
            coef = i / CORRIDOR_POINTS
            x = t_x * (1 - coef) + i_x * coef
            y = t_y * (1 - coef) + i_y * coef
            pygame.draw.circle(canvas, self.background_color, (x, y), SMALL_RADIUS)
        # restore restricted borders
        pygame.draw.rect(canvas, self.restricted_background_color, (0, 0, STATE_BORDER, STATE_H))
        pygame.draw.rect(canvas, self.restricted_background_color, (0, 0, STATE_W, STATE_BORDER))
        pygame.draw.rect(canvas, self.restricted_background_color, (STATE_W - STATE_BORDER, 0, STATE_BORDER, STATE_H))
        pygame.draw.rect(canvas, self.restricted_background_color,
                         (STATE_BORDER, STATE_H - STATE_BORDER, STATE_W, STATE_BORDER))

        self.background_cache = canvas


if __name__ == '__main__':
    player = SimpleHumanPlayer()
    env = DynamicEnv(player=player, render_mode='human')
    env.render()
    done = False

    # game
    while not done:
        action = player.act(123)
        # step
        obs, reward, done, terminated, info = env.step(action)
        env.render()
        print(f"reward: {reward}")

    print(f"Terminated at {env.frame_count} frame\n"
          f"total reward: {env.total_reward}\n"
          f"targets catched: {env.player.target_counter}")
