import gym
from gym.envs.registration import register

class newenv(object):
    def __init__(self, hist_len=0, block_high=0.5, policy_name=None):
        self.hist_len= hist_len
        self.block_high = block_high
        self.policy_name = policy_name
        register(
            id='LunarLanderContinuousPOMDP-v0',
            entry_point='baselines.env.box2d:LunarLanderContinuousPOMDP',
            max_episode_steps=1000,
            reward_threshold=200,
            kwargs={'hist_len':self.hist_len, 'block_high':self.block_high, 'policy_name':self.policy_name}
        )