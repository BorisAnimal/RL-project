"""
Test a trained SAC agent on the SimpleEnv environment
"""

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

from environmetns.env_dynamic import DynamicEnv
from environmetns.env_maze import MazeEnv
from environmetns.env_simple import SimpleEnv
from environmetns.player import SimplePlayer

MODEL_PATH = "models/sac_simple_pretrained_model_20000000_steps.zip"

# Create and wrap the environment
# environmetns = DummyVecEnv([lambda: SimpleEnv(player=SimplePlayer(), time_limit=30, render_mode='human')])
# env = DummyVecEnv([lambda: MazeEnv(player=SimplePlayer(), time_limit=30, render_mode='human')])
env = DummyVecEnv([lambda: DynamicEnv(player=SimplePlayer(), time_limit=30, render_mode='human')])

if __name__ == '__main__':
    # Evaluate the agent
    # Load the trained agent
    model = SAC.load(MODEL_PATH, env=env)
    for i in range(10):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            episode_reward += reward
            if done:
                env.reset()
        print("Episode reward", episode_reward)
        env.render("yes")
