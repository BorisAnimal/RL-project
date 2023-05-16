import os

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

# Create log dir
from environmetns.env_simple import SimpleEnv
from environmetns.env_dynamic import DynamicEnv
from environmetns.env_maze import MazeEnv
from environmetns.player import SimplePlayer

log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)
MODEL_PATH = "models/sac_simple_pretrained_model_20000000_steps.zip"

env = make_vec_env(lambda: Monitor(SimpleEnv(player=SimplePlayer()), log_dir), n_envs=8)

# Create SAC agent
model = SAC.load(MODEL_PATH, env=env,
    verbose=1,
    tensorboard_log=log_dir
)

# Create checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=10_000, save_path=log_dir, name_prefix="sac_dynamic_pretrained_model"
)

# Train the agent
final_model = model.learn(
    total_timesteps=5_000_000,
    callback=[
        checkpoint_callback
    ],
)

final_model.save("sac_dynamic_pretrained_model_final")

if __name__ == '__main__':
    # TODO: check if can parse training logs for plots in report
    pass
