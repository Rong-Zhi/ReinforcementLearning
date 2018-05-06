import gym
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set(color_codes=True)

fig, axes = plt.subplots(3, 3, figsize=(10, 10))
axes = axes.flatten()

for ax in axes:
     ax.clear()

env = gym.make('LunarLanderContinuousPOMDP-v0')
ob = env.reset()
d = {'Pos_x':[], 'Pos_y':[], 'Vel_x':[], 'Vel_y':[], 'Angle':[],
     'Ang_Vel':[], 'Touch_l':[], 'Touch_r':[], 'Block':[]}
observation = pd.DataFrame(data=d)
t = 0
while True:
     env.render()
     act = env.action_space.sample()
     ob, rwd, done, _ = env.step(act)
     print(ob)
     if done:
          break
     t += 1




