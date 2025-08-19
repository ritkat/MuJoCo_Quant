import numpy as np
from Ant_quant import *
env = gym.make('Ant-v5')
model = PPO.load("/home/ritwik/MuJoCo_Quant/logs_ant/best_model_fixscale_1/best_model", env=env)
