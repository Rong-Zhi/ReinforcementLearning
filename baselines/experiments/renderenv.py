import gym
import pandas as pd
import imageio
import os

def get_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

env = gym.make('LunarLanderContinuousPOMDP-v0')
video_path = get_dir('/Users/zhirong/Documents/Masterthesis-code/tmp/LunarLanderContinuousPOMDP-v0/videos')

ob = env.reset()

# d = {'Pos_x':[], 'Pos_y':[], 'Vel_x':[], 'Vel_y':[], 'Angle':[],
#      'Ang_Vel':[], 'Touch_l':[], 'Touch_r':[], 'Block':[]}
# observation = pd.DataFrame(data=d)

frames = []
while True:
     frame = env.render(mode='rgb_array')
     frames.append(frame)
     act = env.action_space.sample()
     ob, rwd, done, _ = env.step(act)
     # print(ob)
     if done:
         imageio.mimsave(video_path + '/' + 'example.gif', frames, fps=20)
         print('Saved video')
         break




