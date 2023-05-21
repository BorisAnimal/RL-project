import os

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv

from environmetns.env_dynamic import DynamicEnv
from environmetns.env_maze import MazeEnv
from environmetns.player import SimplePlayer

# Create log dir
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

################## finetune maze 10M ########################
env = make_vec_env(lambda: Monitor(MazeEnv(player=SimplePlayer()), log_dir), n_envs=16)

# Create SAC agent
model = SAC.load("models/sac_dynamic_pretrained_model_3200000_steps.zip", env)

# Create checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=10_000, save_path=log_dir, name_prefix="sac_maze_pretrained_model"
)

# Train the agent
final_model = model.learn(
    total_timesteps=10_000_000,
    callback=[
        checkpoint_callback
    ]
)

final_model.save("sac_maze_pretrained_model_final")

###################### train dynamic from scratch 20M ######################################
env = make_vec_env(lambda: Monitor(DynamicEnv(player=SimplePlayer()), log_dir), n_envs=16)
# Create SAC agent
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir
)

# Create checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=10_000, save_path=log_dir, name_prefix="sac_dynamic_model"
)

# Train the agent
final_model = model.learn(
    total_timesteps=20_000_000,
    callback=[
        checkpoint_callback
    ],
)

final_model.save("sac_model_dynamic_final")


########################## finetune dynamic 20M ###############################
def create_env():
    env = DynamicEnv(player=SimplePlayer(ray_range=4))
    return Monitor(env, log_dir)


env = make_vec_env(create_env, n_envs=16)

# Create SAC agent
model = SAC.load("models/sac_dynamic_pretrained_model_3200000_steps.zip", env)

# Create checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=10_000, save_path=log_dir, name_prefix="sac_dynamic_pretrained_model"
)

# Train the agent
final_model = model.learn(
    total_timesteps=20_000_000,
    callback=[
        checkpoint_callback
    ],
)

final_model.save("sac_model_dynamic_pretrained_final")


# ########################## finetune dynamic, high penalty ###############################
def create_env():
    env = DynamicEnv(player=SimplePlayer(ray_range=4))
    env.RESTRICTED_ZONE_REWARD = -100
    return Monitor(env, log_dir)


env = make_vec_env(create_env, n_envs=16)

# Create SAC agent
model = SAC(
    "MlpPolicy",
    env,
    # buffer_size=10_000,
    # learning_starts=5_000,
    # batch_size=128,
    verbose=1,
    tensorboard_log=log_dir
)

# Create checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=10_000, save_path=log_dir, name_prefix="sac_dynamic_model_v2"
)

# Train the agent
final_model = model.learn(
    total_timesteps=10_000_000,
    callback=[
        checkpoint_callback
    ],
)

if __name__ == '__main__':
    pass
