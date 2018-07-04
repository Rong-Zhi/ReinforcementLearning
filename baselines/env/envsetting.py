import gym
from gym.envs.registration import register

class newenv(object):
    def __init__(self, hist_len=0, block_high=0.5, version0=True, give_state=True):
        self.hist_len = hist_len
        self.block_high = block_high
        self.version0 = version0
        self.give_state = give_state

        if self.version0:
            register(
                id='LunarLanderContinuousPOMDP-v0',
                entry_point='baselines.env.box2d:LunarLanderContinuousPOMDPv0',
                max_episode_steps=1000,
                reward_threshold=200,
                kwargs={'hist_len':self.hist_len, 'block_high':self.block_high}
            )
        else:
            register(
                id='LunarLanderContinuousPOMDP-v0',
                entry_point='baselines.env.box2d:LunarLanderContinuousPOMDP',
                max_episode_steps=1000,
                reward_threshold=200,
                kwargs={'hist_len':self.hist_len, 'block_high':self.block_high, 'give_state':self.give_state}
            )
