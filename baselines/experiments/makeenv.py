# from baselines.env.box2d.lunar_lander_pomdp import LunarLanderContinuousPOMDP
from baselines import bench, logger
from baselines.env.envsetting import newenv
import gym


newenv(hist_len=5, block_high=5/8)
env = gym.make('LunarLanderContinuousPOMDP-v0')
# env = LunarLanderContinuousPOMDP(hist_len=0)
env = bench.Monitor(env, logger.get_dir())
obs,state = env.reset()
ob_shapce = env.observation_space
total_shape = env.total_space
print("obs:{0}, state:{1}".format(obs, state))
print("total space:", total_shape, total_shape.shape)
print("obs space:", ob_shapce, ob_shapce.shape)


l = 0
# while True:
#     env.render(mode="rgb_array")
#     ac = env.action_space.sample()
#     [obs,state], rwd, done, _ = env.step(ac)
#     print("obs:{0}, state:{1}".format(obs, state))
#     if done:
#         break
#     l+=1
# print("Episode Length from baselines:{}".format(l))