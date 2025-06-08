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
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.distributions import DiagGaussianDistribution

class QuantizedDistribution(DiagGaussianDistribution):
    """
    Custom distribution class that uses quantized actions.
    """
    def __init__(self, latent_dim, action_dim, cfg):
        super().__init__(latent_dim, action_dim)
        self.cfg = cfg

    def 

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0):
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :return:
        """
        mean_actions = nn.Sequential(nn.Linear(latent_dim, self.action_dim))
        # TODO: allow action dependent std
        log_std = nn.Parameter(th.ones(self.action_dim) * log_std_init, requires_grad=True)
        return mean_actions, log_std
    
    # def 

class QuantizedAntFinal(nn.Module):
    def __init__(self, model):
        super().__init__()
        # self.activation = nonlinearity(cfg)
        self.activation = nn.ReLU()

        self.act1 = QuantAct()
        self.fc1 = QuantLinear()

        self.fc1.set_param(model[0])
        self.fc2.set_param(model[2])
        # self.fc3.set_param(model[2])
        

    def forward(self, x, scale_acts=None, scale_weights=None):
        x, act_scaling_factor = self.act1(x, scale_acts, scale_weights)
        x, x_i, fc_scaling_factor = self.fc1(x, act_scaling_factor, name="")
        x, act_scaling_factor1 = self.act2(x, act_scaling_factor, fc_scaling_factor)
        x, x_i, fc_scaling_factor1 = self.fc2(x, act_scaling_factor1)

class QuantizedAntPolicy(nn.Module):
    def __init__(self, model):
        super().__init__()
        # self.activation = nonlinearity(cfg)
        self.activation = nn.ReLU()

        self.act1 = QuantAct()
        self.fc1 = QuantLinear()
        self.act2 = QuantAct(quant_mode="asymmetric")
        self.fc2 = QuantLinear()
        self.act3 = QuantAct(quant_mode="asymmetric")
        self.fc3 = QuantLinear()

        self.fc1.set_param(model[0])
        self.fc2.set_param(model[2])
        # self.fc3.set_param(model[2])
        

    def forward(self, x, scale_acts=None, scale_weights=None):
        x, act_scaling_factor = self.act1(x, scale_acts, scale_weights)
        x, x_i, fc_scaling_factor = self.fc1(x, act_scaling_factor, name="")
        x, act_scaling_factor1 = self.act2(x, act_scaling_factor, fc_scaling_factor)
        x, x_i, fc_scaling_factor1 = self.fc2(x, act_scaling_factor1)
        # x, act_scaling_factor2 = self.act3(x, act_scaling_factor1, fc_scaling_factor1)
        # x = self.fc3(x)
        return x, act_scaling_factor1, fc_scaling_factor1


class QuantizedMlpExtractor(MlpExtractor):
    def __init__(self, feature_dim, net_arch, activation_fn, device="auto"):
        # Call parent constructor to build all attributes
        super().__init__(feature_dim, net_arch, activation_fn, device)

        feed_forward = nn.Sequential(
            nn.Linear(feature_dim, 128),      # model[0]
            nn.ReLU(),            # model[1]
            nn.Linear(128, 128), 
            # model[2]
        )
        # Replace or wrap the networks with quantized versions
        self.policy_net = QuantizedAntPolicy(feed_forward)
        self.value_net = feed_forward

    def forward_actor(self, features: th.Tensor):
        return self.policy_net(features)

class QuantizedActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, cfg, **kwargs):
        self.cfg = cfg

        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build_mlp_extractor(self):
        # Use quantized networks here

        self.mlp_extractor = QuantizedMlpExtractor(
            self.features_dim,
            net_arch= dict(pi=[128, 128], vf=[128, 128]),
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi, act_scaling, fc_scaling = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi, act_scaling, fc_scaling)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

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




