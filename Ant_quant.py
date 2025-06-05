# Write a python script to train a MuJoCo ant agent using the PPO algorithm.
import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from Quantization_utils.quant_modules import *
from Quantization_utils.quant_utils import *
from stable_baselines3.common.callbacks import EvalCallback


class QuantizedAntPolicy(nn.Module):
    def __init__(self, model):
        super().__init__()
        # self.activation = nonlinearity(cfg)
        self.activation = nn.ReLU()

        self.act1 = QuantAct()
        self.fc1 = QuantLinear()
        self.act2 = QuantAct()
        self.fc2 = QuantLinear()
        self.act3 = QuantAct()
        self.fc3 = QuantLinear()

    def forward(self, x, scale_acts=None, scale_weights=None):
        x, _ = self.act1(x, scale_acts, scale_weights)
        x = self.fc1(x)
        x, _ = self.act2(x, scale_acts, scale_weights)
        x = self.fc2(x)
        x, _ = self.act3(x, scale_acts, scale_weights)
        x = self.fc3(x)
        return x

class QuantizedActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, cfg, **kwargs):
        self.cfg = cfg
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build_mlp_extractor(self):
        # Use quantized networks here

        obs_dim = self.observation_space.shape[0]
        act_dim = self.action_space.shape[0]


        feed_forward = nn.Sequential(
            nn.Linear(obs_dim, 128),      # model[0]
            nn.ReLU(),            # model[1]
            nn.Linear(128, act_dim)       # model[2]
        )

        policy_net = QuantizedAntPolicy(feed_forward)  # You can pass a real model if needed
        value_net = feed_forward   # Or define a different one if needed

        # This sets up the extractor properly
        self.mlp_extractor = nn.Module()
        self.mlp_extractor.policy_net = policy_net
        self.mlp_extractor.value_net = value_net

        # Set required attributes so SB3 knows the output dimensions
        self.latent_dim_pi = act_dim
        self.latent_dim_vf = 1

def main():
    # Create the MuJoCo ant environment
    env = gym.make('Ant-v5')
    eval_env = gym.make('Ant-v5')

    # Create a dummy config for quantization (customize as needed)
    cfg = {
        "quant_act": True,
        "quant_weights": True,
        "activation": "relu",
    }

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model/",
        log_path="./logs/results/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )


    # Define PPO agent with your quantized policy
    model = PPO(
        policy=QuantizedActorCriticPolicy,
        env=env,
        policy_kwargs={"cfg": cfg},
        verbose=1
    )

    # Train the agent
    model.learn(total_timesteps=1000000, callback=eval_callback)

    # Save the trained model
    model.save("ppo_ant_quant")

    # Test the trained agent
    obs, _ = eval_env.reset()
    obs, _ = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()

if __name__ == "__main__":
    main()




