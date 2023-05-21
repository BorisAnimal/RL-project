"""
Test a trained SAC agent on the SimpleEnv environment
"""
import sys

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from environmetns.env_dynamic import DynamicEnv
from environmetns.env_maze import MazeEnv
from environmetns.env_simple import SimpleEnv
from environmetns.player import SimplePlayer

if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise Exception("Expected environment name as argument. Exit..")
    env_name = sys.argv[1]

    if env_name == 'simple':
        MODEL_PATH = "models/sac_simple_pretrained_model_20000000_steps.zip"
        env = DummyVecEnv([lambda: SimpleEnv(player=SimplePlayer(), time_limit=30, render_mode='human')])
    elif env_name == 'maze':
        MODEL_PATH = "models/sac_maze_pretrained_model_9760000_steps.zip"
        env = DummyVecEnv([lambda: MazeEnv(player=SimplePlayer(), time_limit=30, render_mode='human')])
    elif env_name == 'dynamic':
        MODEL_PATH = "models/sac_dynamic_pretrained_model_3200000_steps.zip"
        env = DummyVecEnv([lambda: DynamicEnv(player=SimplePlayer(), time_limit=30, render_mode='human')])
    else:
        raise Exception(f'Unexpected environment with name {env_name}. Please, choose "simple", "maze" or "dynamic"')
    # Evaluate the agent
    model = SAC.load(MODEL_PATH, env=env)
    for i in range(300):
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
