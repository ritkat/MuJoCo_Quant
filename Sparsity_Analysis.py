import gymnasium as gym
from stable_baselines3 import PPO  # or whatever algorithm you used
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

# Load the environment
env = gym.make("Ant-v5")
bit_width=8

# Load the trained model
model = PPO.load("logs/best_model/best_model.zip", env=env)
Policy_net = model.policy.mlp_extractor.policy_net
scale_max = Policy_net.act1.x_max.item() 
scale_min = Policy_net.act1.x_min.item() 
# Run inference for 1000 timesteps
obs = env.reset()
obs_check = np.round(obs[0]/(scale_max - scale_min)/(2**(bit_width-1) - 1))
scale = (scale_max - scale_min)/(2**(bit_width-1) - 1)
input_quant = []
print(obs_check)
for t in range(1000):
    if t == 0:
        obs = obs[0]
    action, _ = model.policy.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)[:-1]
    obs_check = np.round(obs/scale)
    input_quant.append(np.round(obs_check/scale))
    # env.render()  # Comment this out if you don't want a window to pop up
    if done:
        obs = env.reset()
