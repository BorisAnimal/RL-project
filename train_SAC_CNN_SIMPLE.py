import os

import numpy as np
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv

from simple_env import SimpleEnv
from simple_player import SimplePlayer

# Create log dir
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

env = make_vec_env(lambda: Monitor(SimpleEnv(player=SimplePlayer()), log_dir), n_envs=16)
# env = VecFrameStack(env, n_stack=5)


# Create SAC agent
model = SAC(
    "MlpPolicy",
    env,
    # buffer_size=10_000,
    # learning_starts=5_000,
    # batch_size=128,
    verbose=1,
    tensorboard_log=log_dir,
    policy_kwargs=dict(normalize_images=False)
)

# Create checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=10000, save_path=log_dir, name_prefix="sac_cnn_model_simple_v2"
)

# Train the agent
final_model = model.learn(
    total_timesteps=20_000_000,
    callback=[
        checkpoint_callback
    ],
)

final_model.save("sac_model_simple_final_v2")


# Create PPO agent
model = PPO(
    "MlpPolicy",
    env,
    # buffer_size=10_000,
    # learning_starts=5_000,
    # batch_size=128,
    ent_coef=0.1,
    verbose=1,
    tensorboard_log=log_dir,
    policy_kwargs=dict(normalize_images=False)
)

# Create checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=10000, save_path=log_dir, name_prefix="ppo_cnn_model_simple_v2"
)

# Train the agent
final_model = model.learn(
    total_timesteps=20_000_000,
    callback=[
        checkpoint_callback
    ],
)

final_model.save("ppo_model_simple_final_v2")

if __name__ == '__main__':
    pass
