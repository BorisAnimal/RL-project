from math import sqrt

from environmetns.env_simple import SimpleEnv, STATE_BORDER, STATE_W, STATE_H
from environmetns.player import SimpleHumanPlayer

WALL_W = 8
WALL_H = 16


class MazeEnv(SimpleEnv):

    def static_background(self):
        # Background and borders
        canvas = SimpleEnv.static_background(self)
        # Maze
        diag = sqrt(STATE_W ** 2 + STATE_H ** 2)
        # quarter circles
        self._draw_wall(canvas, STATE_BORDER, STATE_BORDER, WALL_W, WALL_H)
        self._draw_wall(canvas, int(STATE_W / 2 - STATE_BORDER), STATE_BORDER * 2, WALL_W, WALL_H)
        self._draw_wall(canvas, (STATE_W - STATE_BORDER - WALL_H), STATE_BORDER + WALL_H, WALL_H, WALL_W)
        self._draw_wall(canvas, STATE_BORDER, int(STATE_W / 2), WALL_H, WALL_W)
        self._draw_wall(canvas, STATE_BORDER + WALL_H, STATE_H - STATE_BORDER - WALL_H, WALL_W, WALL_H)
        self._draw_wall(canvas, STATE_W - STATE_BORDER * 4, STATE_H - STATE_BORDER * 3 - WALL_W, WALL_H,
                        WALL_W)

        return canvas

    def _draw_wall(self, canvas, x1, y1, w, h):
        canvas.fill(
            self.restricted_background_color,
            # left,top,w,h
            rect=(x1, y1, w, h)
        )


if __name__ == '__main__':
    player = SimpleHumanPlayer()
    env = MazeEnv(player=player, render_mode='human')
    env.render()
    done = False

    # game
    while not done:
        action = player.act(123)
        # step
        obs, reward, done, terminated, info = env.step(action)
        env.render()

    print(f"Terminated at {env.frame_count} frame")
