# first get familiar with breakout playground

import gym
import numpy as np
from matplotlib import pyplot as plt

env = gym.envs.make('Breakout-v0')
# env.reset()
# while True:
#     action = env.action_space.sample()
#     next_state, reward, done, _ = env.step(action)
#     if done:
#         break
#     env.render()


print("Action space size: {}".format(env.action_space.n))
print(env.unwrapped.get_action_meanings())

observation = env.reset()
print("Observation space shape: {}".format(observation.shape))

