from stable_baselines3 import PPO
# from utils import *
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym

# env = QDAntBulletEnv() 
env = gym.make("Ant-v5")
model = PPO.load("/home/ritwik/MuJoCo_Quant/logs/PPO_normal/best_model/best_model.zip", env=env)

num_runs = 5
max_timesteps = 1000

observations = []
actions = []

for i in range(num_runs):
    obs = env.reset()[0]
    finish = False  # reset finish for each run
    t = 0
    observations_temp = []
    actions_temp = []
    rew = 0

    while not finish:
        observations_temp.append(obs)
        action, _states = model.predict(obs)
        actions_temp.append(action)
        obs, reward, done, truncated,info  = env.step(action)
        rew+=reward
        t += 1
        if t == max_timesteps:
            finish = True
    print(rew)
    observations.append(np.array(observations_temp))  # shape = (1000, k)
    actions.append(np.array(actions_temp))            # shape = (1000, action_dim)

# Convert to array of shape (num_runs, max_timesteps, obs_dim)
observations = np.array(observations)  # shape = (5, 1000, k)
actions = np.array(actions)            # shape = (5, 1000, action_dim)

print("Observations shape:", observations.shape)
print("Actions shape:", actions.shape)

obs_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]  # first five observation dimensions# obs_indices_2 = [5,6,7,8]
obs_indices = np.arange(0,28)
# obs_indices_3 = [9,10,11,12,13]
# obs_indices_4 = [14,15,16,17,18]
# obs_indices_5 = [19,20,21,22,23]
colors = ["blue", "orange", "green", "red", "purple"]

plt.figure(figsize=(12,12))
timesteps = np.arange(0,max_timesteps)
offset = 3.0  # vertical spacing between signals (tune this)
# for i, (idx, color) in enumerate(zip(obs_indices, colors)):
for i, idx in enumerate(obs_indices):
    runs_obs = observations[:, :, idx]  # shape = (5, 1000)
    mean_obs = runs_obs.mean(axis=0)
    std_obs = runs_obs.std(axis=0)

    # Apply vertical offset for "EEG style"
    y_mean = mean_obs + i * offset
    y_lower = mean_obs - std_obs + i * offset
    y_upper = mean_obs + std_obs + i * offset

    plt.plot(timesteps, y_mean, label=f"Mean obs[{idx}]")
    plt.fill_between(timesteps, y_lower, y_upper, alpha=0.3)

plt.xlabel("Timestep")
plt.ylabel("Observation value (offset like EEG)")
plt.title("Observations across 5 runs (EEG-style offset)")
plt.savefig("fix_scale_analysis/ant_observations_eeg_style.png")
# plt.legend()
plt.show()
plt.close()