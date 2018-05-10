from baselines.env.lunar_lander_pomdp import LunarLanderContinuousPOMDP
env = LunarLanderContinuousPOMDP(hist_len=5)
print(env.reset())