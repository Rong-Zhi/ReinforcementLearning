import gym
from gym.envs.registration import register

class newenv(object):
    def __init__(self, hist_len=0):
        self.hist_len= hist_len
        register(
            id='LunarLanderContinuousPOMDP-v0',
            entry_point='baselines.env.box2d:LunarLanderContinuousPOMDP',
            max_episode_steps=1000,
            reward_threshold=200,
            kwargs={'hist_len':self.hist_len}
        )