# Write a python script to train a MuJoCo ant agent using the PPO algorithm.
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from Quantization_utils.quant_modules import *
from Quantization_utils.quant_utils import *
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.preprocessing import get_action_dim
from functools import partial
from typing import Callable, Dict, Any, Optional
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
import imageio
from datetime import datetime
import os
from env_utils import AnymalEnv
from importlib_resources import files               # stdlib >=3.9

os.environ["CUDA_VISIBLE_DEVICES"] = ""

def main():

    env = gym.make(
        "Ant-v5",               # “rgb_array” for headless
    )
    model = PPO(
        policy=ActorCriticPolicy,
        env=env,
        verbose=1,
        tensorboard_log="./ppo_tensorboard_normal/"
    )
    eval_callback = EvalCallback(
        env,
        best_model_save_path="./logs/PPO_normal/best_model/",
        log_path="./logs/PPO_normal/results/128_128/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    obs, info = env.reset()
    model.learn(total_timesteps=10000000, callback=eval_callback)
    frames = []
    for _ in range(2_000):                    # quick smoke-test
        action = env.action_space.sample()    # replace with your policy
        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
        frames.append(frame)
        print(reward)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
    video_path = f"env_trial_change.mp4"
    imageio.mimsave(video_path, frames, fps=30)
    print(f"Saved video to {video_path}")
    # np.save("animal_reward.npy",np.array(avg))

if __name__ == "__main__":
    main()