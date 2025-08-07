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
from mujoco_menagerie import anybotics_anymal_c

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
now = datetime.now().strftime("%Y%m%d_%H%M%S")

Schedule = Callable[[float], float]

def evaluate_policy_manual(model, env, n_episodes=5):
    all_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, done, _ = env.step(action)
            done = terminated
            total_reward += reward
            steps += 1

        all_rewards.append(total_reward)
        print(f"[Eval] Episode {ep+1}: Reward = {total_reward}")

    avg_reward = sum(all_rewards) / len(all_rewards)
    print(f"\n[Eval] Average Reward over {n_episodes} episodes: {avg_reward}")
    return avg_reward


class QuantizedDistribution(DiagGaussianDistribution):
    """
    Custom distribution class that uses quantized actions.
    """
    def __init__(self, action_dim):
        super().__init__(action_dim)
    # def 

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
        mean_actions = QuantizedAntFinal(mean_actions)
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
        self.act2 = QuantAct(quant_mode="symmetric")
        # self.fc2.set_param(model[2])
        # self.fc3.set_param(model[2])

    def forward(self, x, scale_acts=None, scale_weights=None):
        x, act_scaling_factor = self.act1(x, scale_acts, scale_weights)
        x, x_i, fc_scaling_factor = self.fc1(x, act_scaling_factor, name="")
        x, act_scaling_factor1 = self.act2(x, act_scaling_factor, fc_scaling_factor)

        return x

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

def make_proba_distribution_quant(
    action_space: spaces.Space, use_sde: bool = False, dist_kwargs: Optional[Dict[str, Any]] = None
):
    """
    Return an instance of Distribution for the correct type of action space

    :param action_space: the input action space
    :param use_sde: Force the use of StateDependentNoiseDistribution
        instead of DiagGaussianDistribution
    :param dist_kwargs: Keyword arguments to pass to the probability distribution
    :return: the appropriate Distribution object
    """
    if dist_kwargs is None:
        dist_kwargs = {}

    if isinstance(action_space, spaces.Box):
        cls = QuantizedDistribution
        return cls(get_action_dim(action_space), **dist_kwargs)

class QuantizedMlpExtractor(MlpExtractor):
    def __init__(self, feature_dim, net_arch, activation_fn, device="auto"):
        # Call parent constructor to build all attributes
        super().__init__(feature_dim, net_arch, activation_fn, device)

        feed_forward = nn.Sequential(
            nn.Linear(feature_dim, 128),      # model[0]
            nn.ReLU(),            # model[1]
            nn.Linear(128, 128), 
            nn.ReLU()
        )
        # Replace or wrap the networks with quantized versions
        self.policy_net = QuantizedAntPolicy(feed_forward)
        self.value_net = feed_forward

    def forward_actor(self, features: th.Tensor):
        return self.policy_net(features)

class QuantizedActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule,use_sde, cfg, **kwargs):
        self.cfg = cfg
        super().__init__(observation_space, action_space, lr_schedule,use_sde, **kwargs)
        dist_kwargs = None
        self.action_dist = make_proba_distribution_quant(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)
        self.share_features_extractor = False
        self._build(lr_schedule)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, act_scaling: th.Tensor = None, fc_scaling: th.Tensor = None):
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)

    def _build_mlp_extractor(self):
        # Use quantized networks here

        self.mlp_extractor = QuantizedMlpExtractor(
            self.features_dim,
            net_arch= dict(pi=[128, 128], vf=[128, 128]),
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]

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
    
    def evaluate_actions(self, obs: PyTorchObs, actions: th.Tensor):
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi, act_scaling, fc_scaling = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

def main():
         # the Menagerie sub-package
    # 2.  Locate the MJCF; Menagerie ships it with the Python wheel
    #     Prefer the 'scene.xml' if present (includes ground plane); fall back to 'anymal_c.xml'.
    model_dir = files(anybotics_anymal_c)
    xml_path  = model_dir / "scene.xml"
    # if not xml_path.exists():
    # xml_path = model_dir / "anymal_c.xml"      

    # 3.  Create a Gymnasium env that uses that model.
    #    * Ant-v5 already provides walking-robot obs/action spaces & reward hooks
    #    * tweak parameters a bit so the dynamics feel reasonable for ANYmal C
    env = gym.make(
        "Ant-v5",
        xml_file=str(xml_path),
        forward_reward_weight=1.0,            # reward for making forward progress
        ctrl_cost_weight=0.05,                # penalise large torques (ANYmal motors are beefy)
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        main_body=1,                          # main body geom index (torso)
        healthy_z_range=(0.25, 0.9),          # terminate if it tips over / jumps silly high
        reset_noise_scale=0.05,               # add a bit of start-state noise
        frame_skip=25,                        # 25 × 0.002 s MJCF dt  ⇒ 50 ms control step
        max_episode_steps=1_000,
        include_cfrc_ext_in_observation=True,
        exclude_current_positions_from_observation=False,
        render_mode="rgb_array"                 # “rgb_array” for headless
    )
    cfg = {
        "quant_act": True,
        "quant_weights": True,
        "activation": "relu",
    }
    model = PPO(
        policy=QuantizedActorCriticPolicy,
        env=env,
        verbose=1,
        learning_rate=5e-5,
        vf_coef=0.5,
        # clip_range_vf=0.2,
        policy_kwargs={"cfg": cfg},
        tensorboard_log="./ppo_tensorboard/"
    )
    eval_callback = EvalCallback(
        env,
        best_model_save_path="./logs/best_model_quant_alpha_0.75/",
        log_path="./logs/results/128_128/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    obs, info = env.reset()
    model = PPO.load("/home/ritwik/MuJoCo_Quant/logs/best_model_quant_alpha_0.75/best_model.zip", env=env)
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




