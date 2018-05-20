from baselines.env.box2d.lunar_lander_pomdp import LunarLanderContinuousPOMDP
from baselines import bench, logger

import gym
# env = gym.make('LunarLanderContinuousPOMDP-v0')

env = LunarLanderContinuousPOMDP(hist_len=0)
env = bench.Monitor(env, logger.get_dir())
obs = env.reset()
l = 0
while True:
    ac = env.action_space.sample()
    obs, rwd, done, _ = env.step(ac)
    print(done)
    if done:
        break
    l+=1
print("Episode Length from baselines:{}".format(l))