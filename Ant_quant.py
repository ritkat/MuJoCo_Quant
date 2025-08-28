# Write a python script to train a MuJoCo ant agent using the PPO algorithm.
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from Quantization_utils.quant_modules import *
from Quantization_utils.quant_utils import *
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.preprocessing import get_action_dim
from functools import partial
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from typing import Callable, Dict, Any, Optional
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
import imageio
from datetime import datetime
import matplotlib.pyplot as plt
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
now = datetime.now().strftime("%Y%m%d_%H%M%S")

FIX_SCALE = True
HIDDEN_SIZE = 256

Schedule = Callable[[float], float]

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
        if FIX_SCALE==True:
            self.act2 = QuantAct(quant_mode="symmetric", act_range_momentum=-1)
        else:
            self.act2= QuantAct(quant_mode="symmetric")
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
        if FIX_SCALE==True:
            self.act1 = QuantAct(act_range_momentum=-1)
        else:
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
        x = self.activation(x)
        x, act_scaling_factor1 = self.act2(x, act_scaling_factor, fc_scaling_factor)
        x, x_i, fc_scaling_factor1 = self.fc2(x, act_scaling_factor1)
        x = self.activation(x)
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
            nn.Linear(feature_dim, HIDDEN_SIZE),      # model[0]
            nn.ReLU(),            # model[1]
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), 
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
            net_arch= dict(pi=[HIDDEN_SIZE, HIDDEN_SIZE], vf=[HIDDEN_SIZE, HIDDEN_SIZE]),
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
    
    def predict(self, obs: th.Tensor, deterministic: bool = True):

        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        with th.no_grad():
            pi_features = th.tensor(obs, dtype=th.float32)
            latent_pi, act_scaling, fc_scaling = self.mlp_extractor.forward_actor(pi_features)
            distribution = self._get_action_dist_from_latent(latent_pi, act_scaling, fc_scaling)
            actions = distribution.get_actions(deterministic=deterministic)
            log_prob = distribution.log_prob(actions)
            actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions
    
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
    # Create the MuJoCo ant environment
    if FIX_SCALE==True:
        fs = "fixscale"
    else:
        fs = "nonfixscale"
    hs = str(HIDDEN_SIZE)
    model_path = f"./logs_ant_new/_{fs}_{hs}_{now}/"
    os.makedirs(f"./logs_ant_new/_{fs}_{hs}_{now}/", exist_ok=True)
    env = gym.make('Ant-v5')
    eval_env = gym.make('Ant-v5')

    env = make_vec_env('Ant-v5', 
                       n_envs=8, 
                       vec_env_cls=SubprocVecEnv)
    
    # eval_env = make_vec_env('Ant-v5', 
    #                    n_envs=1,
    #                    vec_env_cls=SubprocVecEnv)

    # Create a dummy config for quantization (customize as needed)
    cfg = {
        "quant_act": True,
        "quant_weights": True,
        "activation": "relu",
    }

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_path+"best_model/",
        log_path=model_path+"results/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )


    # Define PPO agent with your quantized policy
    model = PPO(
        policy=QuantizedActorCriticPolicy,
        env=env,
        policy_kwargs={"cfg": cfg},
        verbose=1,
        tensorboard_log="./ppo_tensorboard_ant/trail/"
    )

    # Train the agent
    total_iterations = 70
    timesteps_per_iter = 80_000
    iter_save = 1
    reward_mean = []
    reward_std = []

    for iter in range(total_iterations):
        print(f"\n=== Iteration {iter + 1}/{total_iterations} ===")
        
        # Train for 10,000 timesteps
        model.learn(total_timesteps=timesteps_per_iter,
                    reset_num_timesteps=False,    # continue from previous timestep count
                    tb_log_name=hs+"_"+fs+str(now))
        
        # Save model
        if iter%iter_save == 0:
            # model_path = f"Quant_ant_{now}"
            model.save(model_path+"Quant_ant_"+str(iter))
            np.savez(model_path+"reward_over_time_ant_"+str(now)+".npz", mean=np.array(reward_mean), std = np.array(reward_std))
            print(f"Model saved to {model_path}")
            plt.plot(np.load(model_path+"/reward_over_time_ant_"+str(now)+".npz")['mean'])
            plt.xlabel('Evaluation Iteration (x'+str(timesteps_per_iter*iter_save)+' timesteps)')
            plt.ylabel('Mean Reward over 5 episodes')
            plt.grid()
            plt.savefig(model_path+"/reward_plot_ant_"+str(iter)+"_"+str(now)+".png")
        # --- Manual Evaluation (5 episodes) ---
        # if iter>=300:
        #     env = QDAntBulletEnv_grav()
        #     model.set_env(env)
        #     obs = env.reset()
        # else:
        # env = Monitor(QDAntBulletEnv_grav())
        total_rewards = []
        
        for ep in range(5):
            obs = eval_env.reset()[0]
            done = False
            ep_reward = 0
            t = 0
            while not done:
                action = model.policy.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = eval_env.step(action.reshape(-1,).cpu().numpy())
                ep_reward += reward
                t +=1 
                if t == 1000:
                    done = True
                # env.render()
            print(f"Episode {ep + 1} Reward: {ep_reward} Steps: {t}")
            total_rewards.append(ep_reward)
        reward_mean.append(np.mean(total_rewards))
        reward_std.append(np.std(total_rewards))  
        

    # Save the trained model
    model.save("ppo_ant_quant")

    # if args.env == "Ant":
    render_env = gym.make("Ant-v5", render_mode="rgb_array")
        # render_env = TimeAwareWrapper(render_env)
    # if args.env == "HalfCheetah":
    #     render_env = gymx.make("HalfCheetah-v4", render_mode="rgb_array")
    #     render_env = TimeAwareWrapper(render_env)

    obs, _ = render_env.reset()

    frames = []
    done = False
    timestep = 0
    total_reward = 0
    while not done and timestep < 1000:  # limit max steps for video length
        # Add time feature to obs if your env wrapper requires it
        # obs_with_time = np.append(obs, timestep).reshape(1, -1)
        
        # Get action from your model
        action, _ = model.policy.predict(obs, deterministic=True)
        
        obs, reward, terminated, done, info = render_env.step(np.squeeze(action))
        done = terminated 
        frame = render_env.render()  # Render frame as RGB array
        frames.append(frame)
        
        timestep += 1
        total_reward += reward
    print(f"Total reward: {total_reward}, Steps taken: {timestep}")

    render_env.close()

    # Save the video
    video_path = f"render_{now}.mp4"
    imageio.mimsave(video_path, frames, fps=30)
    print(f"Saved video to {video_path}")
    obs, _ = env.reset()
    env.close()

if __name__ == "__main__":
    main()




