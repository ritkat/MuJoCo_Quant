import mujoco
import numpy as np
from tqdm import tqdm

# # Load model
# model = mujoco.MjModel.from_xml_path("mujoco_menagerie/anybotics_anymal_c/scene.xml")
# data = mujoco.MjData(model)

# # Storage
# observations = []
# actions = []

# # Simulation parameters
# duration = 5.0  # seconds
# fps = 30
# n_steps = int(duration * fps)

# for _ in tqdm(range(n_steps)):
#     # --- Sample random action ---
#     ctrl_range = model.actuator_ctrlrange
#     action = np.random.uniform(ctrl_range[:, 0], ctrl_range[:, 1])
#     data.ctrl[:] = action
#     actions.append(action)

#     # --- Step the simulation ---
#     mujoco.mj_step(model, data)

#     # --- PPO-style observation ---
#     obs = np.concatenate([
#         data.qpos[7:],  # skip root pos/orientation
#         data.qvel[:]
#     ])
#     observations.append(obs)

import gym
from gym import spaces
import mujoco
import numpy as np

class AnymalEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path("mujoco_menagerie/anybotics_anymal_c/scene.xml")
        self.data = mujoco.MjData(self.model)

        # Action space (from actuator ctrlrange)
        ctrl_range = self.model.actuator_ctrlrange
        self.action_space = spaces.Box(
            low=ctrl_range[:, 0],
            high=ctrl_range[:, 1],
            dtype=np.float32
        )

        # Observation space (joint positions/velocities)
        obs_dim = len(self.data.qpos[7:]) + len(self.data.qvel[:])
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.render_mode = render_mode
        self.renderer = mujoco.Renderer(self.model) if render_mode == "rgb_array" else None

    def reset(self, seed=None, options=None):
        self.set_state(
            self.init_qpos + self.np_random.normal(scale=0.01, size=self.model.nq),
            self.init_qvel + self.np_random.normal(scale=0.01, size=self.model.nv)
        )
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.data.ctrl[:] = np.clip(action, self.action_space.low, self.action_space.high)
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = False  # or some condition like fall
        truncated = False  # or time limit
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.renderer:
            self.renderer.update_scene(self.data)
            return self.renderer.render()
        else:
            raise NotImplementedError("Render mode not set to 'rgb_array'.")

    def close(self):
        if self.renderer:
            self.renderer.free()

    def _get_obs(self):
        return np.concatenate([self.data.qpos[7:], self.data.qvel[:]])

    def _compute_reward(self):
        # Very simple reward: forward velocity
        return self.data.qvel[0]  # Forward x-axis velocity


# # Save the data
# np.save("observations.npy", np.array(observations))
# np.save("actions.npy", np.array(actions))
# print("Saved observations and actions.")
