from stable_baselines3 import PPO
import numpy as np
import imageio
import gymnasium as gym
from importlib_resources import files
# Import your environment class
from env_utils import AnymalEnv  # Replace with actual path/module
from mujoco_menagerie import anybotics_anymal_c

model_dir = files(anybotics_anymal_c)
xml_path  = model_dir / "scene.xml"
env_any = gym.make(
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
# --- Load trained model ---
model = PPO.load("/home/ritwik/MuJoCo_Quant/logs/best_model_lr_5e5/best_model.zip", env=env_any)

# --- Create environment for rendering ---
env = env_any
obs, _ = env.reset()

frames = []
done = False
total_reward = 0
timestep = 0

while not done and timestep < 1000:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward

    # frame = env.render()
    # print("render")
    # frames.append(frame)

    timestep += 1

env.close()
print(f"Total Reward: {total_reward}")

# --- Save video ---
imageio.mimsave("anymal_inference.mp4", frames, fps=30)
