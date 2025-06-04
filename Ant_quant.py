# Write a python script to train a MuJoCo ant agent using the PPO algorithm.
import gym
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from Quantization_utils import QuantAct, QuantLinear, nonlinearity


class QuantizedAntPolicy(nn.Module):
    def __init__(self, model, cfg):
        super().__init__()
        self.activation = nonlinearity(cfg)

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
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        input_dim = self.features_extractor.features_dim
        output_dim = self.action_net.out_features  # Typically equals action_space.shape[0]
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        feed_forward = nn.Sequential(
        nn.Linear(obs_dim, 128),      # model[0]
        nonlinearity(cfg),            # model[1]
        nn.Linear(128, act_dim)       # model[2]
        )

        self.mlp_extractor.policy_net = QuantizedAntPolicy(input_dim, output_dim, cfg)
        self.mlp_extractor.value_net = QuantizedAntPolicy(input_dim, 1, cfg)

        self._build(lr_schedule)


def main():
    # Create the MuJoCo ant environment
    env = gym.make('Ant-v2')

    # Initialize the PPO agent
    model = PPO('MlpPolicy', env, verbose=1)

    # I also want to customize the architecture and want to add a custom actor
    # and critic network. This can be done by defining a custom policy.
    
    class CustomActorCriticPolicy(ActorCriticPolicy):
        def __init__(self, *args, **kwargs):
            super(CustomActorCriticPolicy, self).__init__(*args, **kwargs)

        def _build_mlp_extractor(self):
            # Here you can customize the MLP architecture

            return super()._build_mlp_extractor()
    # Use the custom policy in the PPO agent
    model = PPO(CustomActorCriticPolicy, env, verbose=1)


    # Train the agent
    model.learn(total_timesteps=100000)

    # Save the trained model
    model.save("ppo_ant")

    # Optionally, you can test the trained agent
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
    env.close()
if __name__ == "__main__":
    main()




