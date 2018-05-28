import gym
import os
import pandas as pd
import imageio
import numpy as np
import joblib
import tensorflow as tf

from baselines import bench, logger
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.env.envsetting import newenv


save_path = '/home/zhi/Documents/ReinforcementLearning/tmp'

hist_len=10
block_hight=5/8
newenv(hist_len=hist_len, block_high=block_hight, policy_name='MlpPolicy')


def make_env():
    env = gym.make('LunarLanderContinuousPOMDP-v0')
    env = bench.Monitor(env, os.path.join(save_path, 'render-result'), allow_early_resets=True)
    return env
env = DummyVecEnv([make_env])
env = VecNormalize(env)

ob = env.reset()

print(env.observation_space)
print(env.action_space)
# d = {'Pos_x':[], 'Pos_y':[], 'Vel_x':[], 'Vel_y':[], 'Angle':[],
#      'Ang_Vel':[], 'Touch_l':[], 'Touch_r':[], 'Block':[]}
# observation = pd.DataFrame(data=d)

frames = []
while True:
    frame = env.unwrapedrender()
    frames.append(frame)
    act = env.action_space.sample()
    ob, rwd, done, _ = env.step(act)
    # print(ob)
    if done:
        imageio.mimsave(save_path + '/' + 'example.gif', frames, fps=20)
        print('Save video')
        break



