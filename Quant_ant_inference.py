import numpy as np
from Ant_quant import *
env = gym.make('Ant-v5')
import torch as th
import zipfile
import imageio
import io
from Ant_quant import *
# model = PPO.load("/home/ritwik/MuJoCo_Quant/logs_ant/best_model_fixscale_1/best_model", env=env)

def forward_pass(obs):
    obs_quant = torch.round(torch.tensor(obs)*2**e0/m0).to(torch.float32)
    obs_quant = torch.clip(obs_quant, -127, 127)
    a1 = torch.round((torch.matmul(obs_quant,w1.T)+b1)*m1/2**e1)
    # a1 = torch.relu(a1)
    a1 = torch.clip(a1, 0, 255)
    a2 = torch.round((torch.matmul(a1,w2.T)+b2)*m2/2**e2)
    # a2 = torch.relu(a2)
    a2 = torch.clip(a2, 0, 255)
    a3 = torch.round((torch.matmul(a2,w3.T)+b3)*m3/2**e3)
    a3 = torch.clip(a3, -127, 127)

    return a3

def batch_frexp_refactored(inputs):
    """
    Decompose the scaling factor into mantissa and twos exponent, ensuring:
    - Mantissa lies in [0, 2^16)
    - Exponent remains meaningful (adjusted to match the scaled mantissa)

    Parameters:
    ----------
    inputs: Tensor
        Scaling factor

    Returns:
    -------
    mantissa: Tensor
    exponent: Tensor
    """
    shape_of_input = inputs.size()

    # Flatten the input tensor to a 1D tensor
    inputs = inputs.view(-1)

    # Decompose into mantissa and exponent
    output_m, output_e = np.frexp(inputs.cpu().numpy())

    # Prepare adjusted mantissa and exponent lists
    tmp_m = []
    tmp_e = []

    # Define the bias to scale mantissa to [0, 2^16)
    scale_factor = 16  # 2^16

    for idx, (m, e) in enumerate(zip(output_m, output_e)):
        # Scale mantissa from [0.5, 1) to [0, 2^16)
        int_m_shifted = int(Decimal(m * (2 ** scale_factor)).quantize(Decimal('1'), rounding=decimal.ROUND_HALF_UP))

        # Adjust the exponent to account for the scaling of the mantissa
        e -= scale_factor
        e= e*-1
        e = max(0, min(255, e))
        # Ensure mantissa and exponent are within valid bounds
        if int_m_shifted >= 2 ** 16:  # Cap mantissa at maximum (65535)
            int_m_shifted = 2 ** 16 - 1

        elif inputs[idx] >= 65535:
            int_m_shifted = 65535
            e = 0

        tmp_m.append(int_m_shifted)
        tmp_e.append(e)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # tensor = torch.tensor(0.).to(device)

    return torch.from_numpy(np.array(tmp_m)).to(device).view(shape_of_input), \
           torch.from_numpy(np.array(tmp_e)).to(device).view(shape_of_input)


model_path = "/home/ritwik/MuJoCo_Quant/logs_ant/best_model_fixscale_2/best_model.zip"
# Open the zip and load policy.pth
with zipfile.ZipFile(model_path, "r") as archive:
    with archive.open("policy.pth", "r") as f:
        state_dict = th.load(io.BytesIO(f.read()), map_location="cpu")

Sx = state_dict['mlp_extractor.policy_net.act1.act_scaling_factor']
print(Sx)
Sw = state_dict['mlp_extractor.policy_net.fc1.fc_scaling_factor']
Sx1 = state_dict['mlp_extractor.policy_net.act2.act_scaling_factor']
Sw1 = state_dict['mlp_extractor.policy_net.fc2.fc_scaling_factor']
# Sx2 = state_dict['mlp_extractor.policy_net.act3.act_scaling_factor']
Sw2 = state_dict['action_net.fc1.fc_scaling_factor']
Sx2 = state_dict['action_net.act1.act_scaling_factor']
Sw3= state_dict['action_net.fc1.fc_scaling_factor'] 
Sx3 = state_dict['action_net.act2.act_scaling_factor']


m0, e0 = batch_frexp_refactored(Sx)
m1, e1 = batch_frexp_refactored(Sw*Sx/Sx1)
m2, e2 = batch_frexp_refactored(Sw1*Sx1/Sx2)
m3, e3 = batch_frexp_refactored(Sx2*Sw2/Sx3)

w1 = state_dict['mlp_extractor.policy_net.fc1.weight_integer']
b1 = state_dict['mlp_extractor.policy_net.fc1.bias_integer']
w2 = state_dict['mlp_extractor.policy_net.fc2.weight_integer']
b2 = state_dict['mlp_extractor.policy_net.fc2.bias_integer']
w3 = state_dict['action_net.fc1.weight_integer']
b3 = state_dict['action_net.fc1.bias_integer']

env = gym.make('Ant-v5')
obs = env.reset()

import imageio

frames = []
for _ in range(10):
    timestep=0
    obs = env.reset()
    total_reward= 0
    done = False
    while not done and timestep < 1000:
        if timestep==0:
            action = forward_pass(obs[0])*Sx3
        else:
            action = forward_pass(obs)*Sx3
        action = torch.tanh(action) * torch.tensor(env.action_space.high)
        action = action.cpu().numpy()
        # if timestep ==0:
        #     action, _ = model.policy.predict(obs[0], deterministic=True)
        # else:
        #     action, _ = model.policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        # frame = env.render()
        # frames.append(frame)
        # print("rendered")
        obs_req = obs
        # break
        done = terminated or truncated
        
        # if done:
            # print(timestep)
        total_reward += reward
        timestep+=1
    print(total_reward)
# video_path = f"env_trial_quant.mp4"
# imageio.mimsave(video_path, frames, fps=30)
# print(f"Saved video to {video_path}")
